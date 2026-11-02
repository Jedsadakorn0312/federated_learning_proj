import flwr as fl
import torch
from torch_geometric.loader import DataLoader
from dataset import DrugDataset
from model import DrugGNN
from client import DrugClient
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.utils.data import ConcatDataset, random_split
from collections import OrderedDict

# Disable messy Flower logs
fl.common.logger.configure(identifier="FLOWER_LOG", filename="flower.log")

# --- CONFIGURATION ---
NUM_CLIENTS = 4  
NUM_ROUNDS = 50
BATCH_SIZE = 64
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Global variables
CURRENT_MU = 0.0 
CURRENT_FILENAME = "temp_model.pth"

# --- 2. DEFINE EXPERIMENTS ---
EXPERIMENTS = [
    {"mu": 0.0,   "label": "FedAvg (Baseline)",   "color": "red",    "style": "--"},
    {"mu": 0.001, "label": "FedProx (Low mu)",    "color": "blue",   "style": "-."},
    {"mu": 0.1,   "label": "FedProx (Med mu)",    "color": "green",  "style": "-"},
    {"mu": 1.0,   "label": "FedProx (High mu)",   "color": "black",  "style": ":"}
]

# --- 3. CUSTOM STRATEGY FOR SAVING MODELS ---
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_parameters is not None and server_round == NUM_ROUNDS:
            print(f"   💾 Saving global model to '{CURRENT_FILENAME}'...")
            params_dict = zip(DrugGNN().state_dict().keys(), fl.common.parameters_to_ndarrays(aggregated_parameters))
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            torch.save(state_dict, CURRENT_FILENAME)

        return aggregated_parameters, aggregated_metrics

# --- 4. DATA PREPARATION (SAVE TO FILES) ---
# ฟังก์ชันนี้จะเตรียมข้อมูลและ "เซฟลงไฟล์" เพื่อแก้ MemoryError
def prepare_data_to_files():
    print(f"🔄 Preparing Data & Saving to Disk (Fixing MemoryError)...")
    
    # Load Datasets
    ds_diabetes = DrugDataset(root='.', csv_file="client_diabetes.csv")
    ds_heart = DrugDataset(root='.', csv_file="client_heart.csv")
    ds_alzheimers = DrugDataset(root='.', csv_file="client_alzheimers.csv")

    # Split Diabetes
    half_len = len(ds_diabetes) // 2
    ds_dia_A, ds_dia_B = random_split(ds_diabetes, [half_len, len(ds_diabetes) - half_len])

    client_map = {
        0: ds_dia_A,
        1: ds_heart,
        2: ds_alzheimers,
        3: ds_dia_B
    }

    # 1. Save Client Data
    for cid, dataset in client_map.items():
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_d, test_d = random_split(dataset, [train_size, test_size])
        
        # ✅ KEY FIX: Save to .pt file
        torch.save((train_d, test_d), f"client_{cid}.pt")
        print(f"   ✅ Saved client_{cid}.pt")

    # 2. Save Centralized Data (Combine all)
    all_train = []
    all_test = []
    for i in range(4):
        tr, te = torch.load(f"client_{i}.pt") # Re-load to ensure consistency
        all_train.append(tr)
        all_test.append(te)
    
    cen_train = ConcatDataset(all_train)
    cen_test = ConcatDataset(all_test)
    torch.save((cen_train, cen_test), "centralized_data.pt")
    print(f"   ✅ Saved centralized_data.pt")

# --- 5. CENTRALIZED TRAINING ---
def run_centralized():
    print("\n🔵 STARTING CENTRALIZED MODEL (The Benchmark)...")
    # ✅ Load from file
    train_d, test_d = torch.load("centralized_data.pt")
    
    train_loader = DataLoader(train_d, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_d, batch_size=BATCH_SIZE, shuffle=False)
    
    model = DrugGNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    
    loss_history = []
    
    for epoch in range(NUM_ROUNDS):
        model.train()
        for data in train_loader:
            data = data.to(DEVICE)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y.view(-1, 1).float())
            loss.backward()
            optimizer.step()
        
        # Eval
        model.eval()
        total_loss = 0
        steps = 0
        with torch.no_grad():
            for data in test_loader:
                data = data.to(DEVICE)
                out = model(data)
                loss = criterion(out, data.y.view(-1, 1).float())
                total_loss += loss.item()
                steps += 1
        avg_loss = total_loss / steps
        loss_history.append(avg_loss)
        
        if (epoch+1) % 10 == 0:
            print(f"   Epoch {epoch+1}: Loss = {avg_loss:.4f}")
            
    torch.save(model.state_dict(), "model_centralized.pth")
    return loss_history

# --- 6. FEDERATED CLIENT SETUP ---
def client_fn(cid: str) -> fl.client.Client:
    # ✅ KEY FIX: Load ONLY my own file. No global variable access.
    train_data, test_data = torch.load(f"client_{cid}.pt")
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    model = DrugGNN().to(DEVICE)
    return DrugClient(model, train_loader, test_loader, DEVICE, mu=CURRENT_MU)

def weighted_average(metrics):
    losses = [num_examples * m["mse"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    if sum(examples) == 0: return {"mse": 0}
    return {"mse": sum(losses) / sum(examples)}

# --- 7. MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Prepare Data First
    prepare_data_to_files()

    results = {}

    # 2. Run Centralized
    cen_loss = run_centralized()
    results["Centralized (Ideal)"] = {
        "rounds": range(1, NUM_ROUNDS + 1), 
        "loss": cen_loss, 
        "color": "cyan", 
        "style": "-"
    }

    # 3. Run Federated Experiments
    print(f"\n🚀 STARTING FEDERATED EXPERIMENTS ({NUM_ROUNDS} Rounds each)...")

    for exp in EXPERIMENTS:
        mu_val = exp["mu"]
        label = exp["label"]
        CURRENT_MU = mu_val
        
        safe_label = label.replace(" ", "_").replace("(", "").replace(")", "").replace("$\mu$", "mu").lower()
        CURRENT_FILENAME = f"model_{safe_label}.pth"
        
        print(f"\n🧪 Running: {label} ...")
        
        strategy = SaveModelStrategy(
            fraction_fit=1.0, 
            fraction_evaluate=1.0,
            min_fit_clients=4,       
            min_evaluate_clients=4, 
            min_available_clients=4,
            evaluate_metrics_aggregation_fn=weighted_average,
        )

        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=NUM_CLIENTS,
            config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
            strategy=strategy,
            client_resources={"num_cpus": 1, "num_gpus": 0.0},
            ray_init_args={"include_dashboard": False} # Fix Ray crash
        )
        
        loss = [data[1] for data in history.losses_distributed]
        results[label] = {
            "rounds": [data[0] for data in history.losses_distributed], 
            "loss": loss, 
            "color": exp["color"], 
            "style": exp["style"]
        }
        print(f"   Finished {label}. Final Loss: {loss[-1]:.4f}")

    # 4. Clean up temp files (Optional)
    for f in ["centralized_data.pt"] + [f"client_{i}.pt" for i in range(4)]:
        if os.path.exists(f): os.remove(f)

    # 5. Plotting
    print("\n📊 Generating Graph...")
    plt.figure(figsize=(12, 8))
    
    cen_data = results["Centralized (Ideal)"]
    plt.plot(cen_data["rounds"], cen_data["loss"], 
             color=cen_data["color"], linestyle=cen_data["style"], linewidth=3, 
             label="Centralized (Ideal)", alpha=0.6)

    for label, data in results.items():
        if label == "Centralized (Ideal)": continue
        plt.plot(data["rounds"], data["loss"], 
                 color=data["color"], linestyle=data["style"], linewidth=2, 
                 label=label, marker='o', markersize=4, markevery=5)

    plt.title('Final Comparison: Centralized vs. FedAvg vs. FedProx (4 Clients)', fontsize=16)
    plt.xlabel('Rounds', fontsize=14)
    plt.ylabel('MSE Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig('experiment_1_comparison.png', dpi=300)
    print("✅ Saved Graph and Models.")
    plt.show()