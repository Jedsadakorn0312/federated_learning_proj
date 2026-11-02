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
import ray

# Disable messy Flower logs
fl.common.logger.configure(identifier="FLOWER_LOG", filename="flower.log")

# --- CONFIGURATION ---
NUM_CLIENTS = 4         # 4 Clients (Diabetes A, Heart, Alz, Diabetes B)
NUM_ROUNDS = 800         # 50 Rounds
BATCH_SIZE = 64
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ============================================================
# 1. PREPARE DATA (SAVE TO FILES TO FIX MEMORY ERROR)
# ============================================================
def prepare_data_to_files():
    print(f"🔄 Preparing Data for 4 Clients (Global Test Set) & Saving to Disk...")
    
    # 1. โหลดข้อมูล Train (80%) ของแต่ละโรค
    ds_diabetes_train = DrugDataset(root='.', csv_file="train_client_diabetes.csv")
    ds_heart_train = DrugDataset(root='.', csv_file="train_client_heart.csv")
    ds_alz_train = DrugDataset(root='.', csv_file="train_client_alzheimers.csv")

    # 2. โหลดข้อมูล Global Test Set (20%) สำหรับให้ทุกคนใช้สอบมาตรฐานเดียวกัน
    global_val_ds = DrugDataset(root='.', csv_file="global_val_set.csv")

    # หั่นครึ่งเบาหวาน (ตามลอจิกเดิมของคุณสำหรับ Ex 2)
    half_len = len(ds_diabetes_train) // 2
    ds_dia_A, ds_dia_B = random_split(ds_diabetes_train, [half_len, len(ds_diabetes_train) - half_len])

    # Map ข้อมูล Train ให้ 4 Clients
    client_train_map = {
        0: ds_dia_A,
        1: ds_heart_train,
        2: ds_alz_train,
        3: ds_dia_B
    }

    # 3. Save Client Data (จับคู่ Local Train + Global Test)
    for cid, train_d in client_train_map.items():
        # 🛑 ไม่ต้องทำ random_split แล้ว! 
        # ให้ Client เทรนด้วย train_d ของตัวเอง แต่เวลาสอบ ให้สอบด้วย global_test_ds
        torch.save((train_d, global_val_ds), f"client_{cid}.pt")
        print(f"   ✅ Saved client_{cid}.pt (Train: {len(train_d)}, Test: {len(global_val_ds)})")

    # 4. Save Centralized Data (เอา Train ทั้งหมดมารวมกัน)
    cen_train = ConcatDataset([ds_diabetes_train, ds_heart_train, ds_alz_train])
    torch.save((cen_train, global_val_ds), "centralized_data.pt")
    print(f"   ✅ Saved centralized_data.pt (Train: {len(cen_train)}, Test: {len(global_val_ds)})")

# ==========================================
# PART 2: CENTRALIZED LEARNING 
# ==========================================
def run_centralized():
    print("\n🔵 STARTING CENTRALIZED LEARNING...")
    
    train_d, test_d = torch.load("centralized_data.pt")
    train_loader = DataLoader(train_d, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_d, batch_size=BATCH_SIZE, shuffle=False)
    
    model = DrugGNN().to(DEVICE)
    
    # กำหนด Optimizer (Weight Decay 1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.000264, weight_decay=5.85e-05)
    
    # กำหนด Scheduler (ลด LR ลงครึ่งนึง ถ้า Loss ไม่ลดลงติดต่อกัน 10 รอบ)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50)
    
    criterion = torch.nn.MSELoss()
    
    loss_history = []
    best_loss = float('inf') 
    
    for epoch in range(NUM_ROUNDS):
        # --- TRAINING ---
        model.train()
        train_loss_total = 0
        train_steps = 0
        for data in train_loader:
            data = data.to(DEVICE)
            optimizer.zero_grad()
            out = model(data)
            
            target = data.y.view(-1, 1).float() / 10.0
            loss = criterion(out, target)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # เก็บค่า Train Loss
            train_loss_total += loss.item()
            train_steps += 1
            
        avg_train_loss = train_loss_total / train_steps
            
        # --- EVALUATION (TEST) ---
        model.eval()
        test_loss_total = 0
        test_steps = 0
        with torch.no_grad():
            for data in test_loader:
                data = data.to(DEVICE)
                out = model(data)
                
                target = data.y.view(-1, 1).float() / 10.0
                loss = criterion(out, target)
                
                test_loss_total += loss.item()
                test_steps += 1
                
        avg_test_loss = test_loss_total / test_steps
        loss_history.append(avg_test_loss)
        
        # 
        # scheduler.step(avg_test_loss)
        
        # พรินต์ทั้ง Train, Test และ Learning Rate มาเทียบกันให้เห็นชัดๆ
        if (epoch+1) % 5 == 0:
            # ดึงค่า LR ปัจจุบันมาแสดงผล
            current_lr = optimizer.param_groups[0]['lr']
            print(f"   Epoch {epoch+1:3d}: Train Loss = {avg_train_loss:.4f} | Test Loss = {avg_test_loss:.4f} | LR = {current_lr:.6f}")
            
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            torch.save(model.state_dict(), "model_centralized_ex_2.pth")
            # print(f"      💾 New Best Model Saved! (Test Loss: {best_loss:.4f})")
            
    print(f"   🏆 Best Test Loss Achieved: {best_loss:.4f}")
    return loss_history

# ==========================================
# PART 3: FEDERATED LEARNING (FedAvg)
# ==========================================

# Custom Strategy to save the model at the end
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_parameters is not None and server_round == NUM_ROUNDS:
            print(f"   💾 Saving global FedAvg model to 'model_fedavg_ex2.pth'...")
            params_dict = zip(DrugGNN().state_dict().keys(), fl.common.parameters_to_ndarrays(aggregated_parameters))
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            torch.save(state_dict, "model_fedavg_ex2.pth")

        return aggregated_parameters, aggregated_metrics

def client_fn(cid: str) -> fl.client.Client:
    # Load from file (Fix Memory Error)
    train_d, test_d = torch.load(f"client_{cid}.pt")
    
    train_loader = DataLoader(train_d, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_d, batch_size=BATCH_SIZE, shuffle=False)
    
    model = DrugGNN().to(DEVICE)
    # mu=0.0 is standard FedAvg
    return DrugClient(model, train_loader, test_loader, DEVICE, mu=0.0)

def weighted_average(metrics):
    losses = [num_examples * m["mse"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    if sum(examples) == 0: return {"mse": 0}
    return {"mse": sum(losses) / sum(examples)}

def run_federated_learning():
    print("\n🔴 STARTING FEDERATED LEARNING (FedAvg - 4 Clients)...")

    if ray.is_initialized():
        ray.shutdown()
        
    strategy = SaveModelStrategy(
        fraction_fit=1.0, fraction_evaluate=1.0,
        min_fit_clients=4, min_evaluate_clients=4, min_available_clients=4,
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
    return loss

# ==========================================
# MAIN COMPARISON
# ==========================================
if __name__ == "__main__":
    # 1. Prepare Data
    prepare_data_to_files()

    # 2. Run Centralized
    centralized_loss = run_centralized()
    
    # 3. Run Federated
    federated_loss = run_federated_learning()
    
    # # 4. Clean up temp files (Optional)
    # for f in ["centralized_data.pt"] + [f"client_{i}.pt" for i in range(4)]:
    #     if os.path.exists(f): os.remove(f)

    # 5. Plot Comparison
    print("\n📊 Generating Comparison Graph...")
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, NUM_ROUNDS + 1)
    
    # Plot Centralized (The Ideal)
    plt.plot(epochs, centralized_loss, 'b--', linewidth=2, label='Centralized (Ideal Upper Bound)')
    
    # Plot Federated (The Reality)
    plt.plot(epochs, federated_loss, 'r-o', linewidth=2, label='Federated Learning (FedAvg)')
    
    plt.title('Performance Gap: Centralized vs. FedAvg (4 Clients)', fontsize=14)
    plt.xlabel('Training Epochs / Rounds', fontsize=12)
    plt.ylabel('Test Loss (MSE)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig('experiment_2_result.png', dpi=300)
    print("✅ Saved comparison graph to 'experiment_2_result.png'")
    print("✅ Saved Models: 'model_centralized.pth', 'model_fedavg.pth'")
    plt.show()