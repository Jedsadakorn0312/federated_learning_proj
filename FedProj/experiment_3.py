import flwr as fl
import torch
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from dataset import DrugDataset
from model import DrugGNN
from client import DrugClient
from torch.utils.data import ConcatDataset, random_split
import numpy as np
import os
import ray # 
from collections import OrderedDict

# Disable messy Flower logs
fl.common.logger.configure(identifier="FLOWER_LOG", filename="flower.log")

# --- CONFIGURATION (Experiment 3: IID / Mixed Data) ---
NUM_CLIENTS = 4        
NUM_ROUNDS = 800
NUM_ROUNDS_FED = 800
BATCH_SIZE = 64
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CLIENT_FILES = ["client_diabetes.csv", "client_heart.csv", "client_alzheimers.csv"]

# ============================================================
# 1. PREPARE DATA (IID & Save to Disk)
# ============================================================
def prepare_data():
    print("🔄 Preparing IID Data (Mixing all diseases with Global Test Set)...")
    
    # 1. โหลดข้อมูล Global Test Set (20% ที่เตรียมไว้ให้ทุกคนใช้สอบเหมือนกัน)
    try:
        global_val_ds = DrugDataset(root='.', csv_file="global_val_set.csv")
        print(f"   📊 Global Test Set: {len(global_val_ds)} samples")
    except Exception as e:
        print(f"❌ Error loading global_test_set.csv: {e}")
        return

    # 2. โหลดไฟล์ Train ของทุกโรคมารวมกัน (80% ที่เหลือ)
    TRAIN_FILES = ["train_client_diabetes.csv", "train_client_heart.csv", "train_client_alzheimers.csv"]
    all_train_datasets = []
    
    for f in TRAIN_FILES:
        try:
            ds = DrugDataset(root='.', csv_file=f)
            all_train_datasets.append(ds)
        except Exception as e:
            print(f"⚠️ Warning: Skip {f} - {e}")

    full_train_dataset = ConcatDataset(all_train_datasets)
    total_train_len = len(full_train_dataset)
    print(f"   📊 Total Train Samples (Mixed): {total_train_len}")

    # 3. Save Centralized Data (เอา Train ทั้งหมดจับคู่กับ Global Test)
    # 🛑 เลิกใช้ random_split หั่น 80/20 ตรงนี้แล้ว เพราะเราแบ่งมาจากไฟล์ CSV แล้ว
    torch.save((full_train_dataset, global_val_ds), "centralized_data_ex3.pt")
    print("   💾 Saved centralized_data.pt")

    # 4. Split Federated Train Data (IID - 4 Equal Parts)
    # แบ่งข้อมูล Train ก้อนใหญ่ ออกเป็น 4 กองเท่าๆ กัน
    part_len = total_train_len // NUM_CLIENTS
    lengths = [part_len] * NUM_CLIENTS
    lengths[-1] += (total_train_len - sum(lengths)) # เศษที่เหลือปัดให้คนสุดท้าย

    # random_split ตรงนี้ทำหน้าที่ "สับไพ่และแบ่งเค้ก" (ผสมโรคให้ IID)
    client_train_partitions = random_split(full_train_dataset, lengths)

    # 5. Save Client Data (จับคู่เค้กส่วนของตัวเอง + ข้อสอบ Global Test)
    for cid, train_partition in enumerate(client_train_partitions):
        # 🛑 เลิกใช้ random_split หั่น 80/20 ตรงนี้เช่นกัน
        torch.save((train_partition, global_val_ds), f"client_data_ex3_{cid}.pt")
        print(f"   💾 Saved client_data_{cid}.pt (Train: {len(train_partition)}, Test: {len(global_val_ds)})")

# ============================================================
# 2. CENTRALIZED LEARNING 
# ============================================================
def run_centralized():
    print("\n🔵 STARTING CENTRALIZED LEARNING...")
    
    train_d, test_d = torch.load("centralized_data_ex3.pt")
    train_loader = DataLoader(train_d, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_d, batch_size=BATCH_SIZE, shuffle=False)
    
    model = DrugGNN().to(DEVICE)
    
    # กำหนด Optimizer (Weight Decay 1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.000416, weight_decay=1.84e-05)
    
    # กำหนด Scheduler (ลด LR ลงครึ่งนึง ถ้า Loss ไม่ลดลงติดต่อกัน 10 รอบ)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=30)
    
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
            torch.save(model.state_dict(), "model_centralized_ex_3.pth")
            # print(f"      💾 New Best Model Saved! (Test Loss: {best_loss:.4f})")
            
    print(f"   🏆 Best Test Loss Achieved: {best_loss:.4f}")
    return loss_history

    # # ✅ Save Model for Ex 3
    # torch.save(model.state_dict(), "model_centralized_ex_3.pth")
    # print("   💾 Saved 'model_centralized_ex_3.pth'")
            
    # return loss_history

# ============================================================
# 3. FEDERATED LEARNING (FedAvg)
# ============================================================

# ✅ Custom Strategy to Save Model
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        # Save at the last round
        if aggregated_parameters is not None and server_round ==NUM_ROUNDS_FED:
            print(f"   💾 Saving global FedAvg model to 'model_fedavg_ex_3.pth'...")
            params_dict = zip(DrugGNN().state_dict().keys(), fl.common.parameters_to_ndarrays(aggregated_parameters))
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            torch.save(state_dict, "model_fedavg_ex_3.pth")

        return aggregated_parameters, aggregated_metrics

def client_fn(cid: str) -> fl.client.Client:
    # Load from file (Fix Memory Error)
    train_d, test_d = torch.load(f"client_data_ex3_{cid}.pt")
    
    train_loader = DataLoader(train_d, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_d, batch_size=BATCH_SIZE, shuffle=False)
    
    model = DrugGNN().to(DEVICE)
    return DrugClient(model, train_loader, test_loader, DEVICE, mu=0.0) # FedAvg

def weighted_average(metrics):
    losses = [num_examples * m["mse"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    if sum(examples) == 0: return {"mse": 0}
    return {"mse": sum(losses) / sum(examples)}

def run_federated():
    print("\n🔴 STARTING FEDERATED LEARNING (IID - 4 Clients)...")

    # ✅ Ray Cleanup: Prevent "Node timed out" error
    if ray.is_initialized():
        ray.shutdown()

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
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS_FED),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
        
        # ✅ FIX: Disable Dashboard + Explicit Resource management
        ray_init_args={
            "include_dashboard": False, 
            "num_cpus": 4, 
            "ignore_reinit_error": True
        } 
    )
    return [data[1] for data in history.losses_distributed]

# ============================================================
# 4. MAIN & PLOT
# ============================================================
import pandas as pd # เพิ่ม import pandas เข้ามาเพื่อใช้จัดการไฟล์ CSV

if __name__ == "__main__":
    # 1. Prepare Data
    prepare_data()

    # 2. Run Experiments
    cen_loss = run_centralized()
    fed_loss = run_federated()
    
    # ==========================================
    # 🌟 NEW: Save Logs to CSV
    # ==========================================
    # เช็คความยาวของข้อมูลเผื่อฝั่งไหนรันไม่ครบ แล้วเติม None ให้ตารางเท่ากัน
    max_len = max(len(cen_loss), len(fed_loss))
    cen_loss_padded = cen_loss + [None] * (max_len - len(cen_loss))
    fed_loss_padded = fed_loss + [None] * (max_len - len(fed_loss))
    
    # สร้างตาราง DataFrame และเซฟเป็น CSV
    df_logs = pd.DataFrame({
        "Round": range(1, max_len + 1),
        "Centralized_Loss": cen_loss_padded,
        "FedAvg_Loss": fed_loss_padded
    })
    df_logs.to_csv("experiment_3_logs.csv", index=False)
    print("   💾 Saved training logs to 'experiment_3_logs.csv'")
    # ==========================================
    
    # 3. Plot Graph
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(cen_loss) + 1)
    
    plt.plot(epochs, cen_loss, 'b--', linewidth=2, label='Centralized (Global Data)')
    
    # เผื่อความยาว fed_loss ไม่เท่า cen_loss จะได้พล๊อตได้ไม่ error
    fed_epochs = range(1, len(fed_loss) + 1)
    plt.plot(fed_epochs, fed_loss, 'r-o', linewidth=2, label='FedAvg (4 Clients - IID Mixed Data)')
    
    plt.title('Performance: IID Setting (Mixed Diseases)', fontsize=14)
    plt.xlabel('Rounds', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig('experiment_3_result.png')
    print("✅ Graph saved to 'experiment_3_result.png'")
    print("✅ Models saved: 'model_centralized_ex_3.pth', 'model_fedavg_ex_3.pth'")
    
    # # 4. Cleanup Temp Files
    # for f in ["centralized_data.pt"] + [f"client_data_{i}.pt" for i in range(NUM_CLIENTS)]:
    #     if os.path.exists(f): os.remove(f)
            
    plt.show()