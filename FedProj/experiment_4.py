import flwr as fl
import torch
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from dataset import DrugDataset
from model import DrugGNN
from client import DrugClient
from torch.utils.data import ConcatDataset, random_split
import pandas as pd
import os
import ray 
from collections import OrderedDict

# Disable messy Flower logs
fl.common.logger.configure(identifier="FLOWER_LOG", filename="flower.log")

# --- CONFIGURATION (Experiment 4: Realistic Non-IID / Quantity Skew) ---
NUM_CLIENTS = 4        
NUM_EPOCHS_CENTRALIZED = 800
NUM_ROUNDS_FED = 800
BATCH_SIZE = 64
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ============================================================
# 1. PREPARE DATA (Realistic Non-IID - Quantity Skew)
# ============================================================
def prepare_data_realistic_non_iid():
    print("🔄 Preparing Realistic Non-IID Data (Quantity Skew) with Global Test Set...")
    
    # 1. โหลดข้อมูล Global Test Set (ข้อสอบกลาง 20%)
    try:
        global_val_ds = DrugDataset(root='.', csv_file="global_val_set.csv")
        print(f"   📊 Global Test Set: {len(global_val_ds)} samples")
    except Exception as e:
        print(f"❌ Error loading global_test_set.csv: {e}")
        return

    # 2. โหลดข้อมูล Train ทีละโรคแยกกัน (80% ที่เหลือ)
    try:
        ds_diabetes_train = DrugDataset(root='.', csv_file="train_client_diabetes.csv")
        ds_heart_train = DrugDataset(root='.', csv_file="train_client_heart.csv")
        ds_alzheimer_train = DrugDataset(root='.', csv_file="train_client_alzheimers.csv")
    except Exception as e:
        print(f"⚠️ Error loading training datasets: {e}")
        return

    # 3. สร้าง Centralized Data (เอา Train ทั้งหมดมารวมกันเป็น Benchmark)
    full_train_dataset = ConcatDataset([ds_diabetes_train, ds_heart_train, ds_alzheimer_train])
    
    # 🛑 เลิกหั่น 80/20 ตรงนี้แล้ว จับคู่ Train ก้อนรวม กับ Global Test ได้เลย
    torch.save((full_train_dataset, global_val_ds), "centralized_data_ex4.pt")
    print(f"   💾 Saved centralized_data_ex4.pt (Train: {len(full_train_dataset)}, Test: {len(global_val_ds)})")

    # --- ฟังก์ชันช่วยหั่นเค้กตามสัดส่วน ---
    def split_by_ratio(dataset, ratios):
        lengths = [int(r * len(dataset)) for r in ratios]
        lengths[-1] = len(dataset) - sum(lengths[:-1]) # ป้องกันเศษทศนิยม
        return random_split(dataset, lengths)

    # 4. กำหนดสัดส่วน [Client 0, Client 1, Client 2, Client 3 (รพ.ทั่วไป)]
    # หั่นเฉพาะข้อมูล Train ให้เอียงไปตามความเชี่ยวชาญของแต่ละโรงพยาบาล
    split_diabetes = split_by_ratio(ds_diabetes_train, [0.60, 0.05, 0.05, 0.30])
    split_heart    = split_by_ratio(ds_heart_train,    [0.05, 0.60, 0.05, 0.30])
    split_alz      = split_by_ratio(ds_alzheimer_train,[0.05, 0.05, 0.60, 0.30])

    # 5. จับชิ้นส่วนมาประกอบร่างให้แต่ละ Client
    client_train_datasets = [
        ConcatDataset([split_diabetes[0], split_heart[0], split_alz[0]]), # Client 0 (ศูนย์เบาหวาน)
        ConcatDataset([split_diabetes[1], split_heart[1], split_alz[1]]), # Client 1 (ศูนย์โรคหัวใจ)
        ConcatDataset([split_diabetes[2], split_heart[2], split_alz[2]]), # Client 2 (ศูนย์อัลไซเมอร์)
        ConcatDataset([split_diabetes[3], split_heart[3], split_alz[3]])  # Client 3 (รพ. ทั่วไป)
    ]

    # 6. บันทึกข้อมูลของแต่ละ Client
    for cid, train_partition in enumerate(client_train_datasets):
        # 🛑 เลิกใช้ random_split หั่น 80/20 ตรงนี้เช่นกัน
        # จับคู่ Local Train (สัดส่วนเอียงๆ) + ข้อสอบ Global Test (ข้อสอบมาตรฐาน)
        torch.save((train_partition, global_val_ds), f"client_data_ex4_{cid}.pt")
        print(f"   💾 Saved client_data_ex4_{cid}.pt (Train: {len(train_partition)}, Test: {len(global_val_ds)})")
# ============================================================
# 2. CENTRALIZED LEARNING 
# ============================================================
def run_centralized():
    print("\n🔵 STARTING CENTRALIZED LEARNING...")
    
    train_d, test_d = torch.load("centralized_data_ex4.pt")
    train_loader = DataLoader(train_d, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_d, batch_size=BATCH_SIZE, shuffle=False)
    
    model = DrugGNN().to(DEVICE)
    
    # กำหนด Optimizer (Weight Decay 1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.000416, weight_decay=1.84e-05)
    
    # กำหนด Scheduler (ลด LR ลงครึ่งนึง ถ้า Loss ไม่ลดลงติดต่อกัน 10 รอบ)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50)
    
    criterion = torch.nn.MSELoss()
    
    loss_history = []
    best_loss = float('inf') 
    
    for epoch in range(NUM_EPOCHS_CENTRALIZED):
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
        scheduler.step(avg_test_loss)
        
        # พรินต์ทั้ง Train, Test และ Learning Rate มาเทียบกันให้เห็นชัดๆ
        if (epoch+1) % 5 == 0:
            # ดึงค่า LR ปัจจุบันมาแสดงผล
            current_lr = optimizer.param_groups[0]['lr']
            print(f"   Epoch {epoch+1:3d}: Train Loss = {avg_train_loss:.4f} | Test Loss = {avg_test_loss:.4f} | LR = {current_lr:.6f}")
            
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            torch.save(model.state_dict(), "model_centralized_ex_4.pth")
            # print(f"      💾 New Best Model Saved! (Test Loss: {best_loss:.4f})")
            
    print(f"   🏆 Best Test Loss Achieved: {best_loss:.4f}")
    return loss_history


# ============================================================
# 3. FEDERATED LEARNING (FedAvg)
# ============================================================
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_parameters is not None and server_round == NUM_ROUNDS_FED:
            print(f"   💾 Saving global FedAvg model to 'model_fedavg_ex_4.pth'...")
            params_dict = zip(DrugGNN().state_dict().keys(), fl.common.parameters_to_ndarrays(aggregated_parameters))
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            torch.save(state_dict, "model_fedavg_ex_4.pth")

        return aggregated_parameters, aggregated_metrics

def client_fn(cid: str) -> fl.client.Client:
    train_d, test_d = torch.load(f"client_data_ex4_{cid}.pt")
    train_loader = DataLoader(train_d, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_d, batch_size=BATCH_SIZE, shuffle=False)
    model = DrugGNN().to(DEVICE)
    return DrugClient(model, train_loader, test_loader, DEVICE, mu=0.0) 

def weighted_average(metrics):
    losses = [num_examples * m["mse"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    if sum(examples) == 0: return {"mse": 0}
    return {"mse": sum(losses) / sum(examples)}

def run_federated():
    print("\n🔴 STARTING FEDERATED LEARNING (Realistic Non-IID - 4 Clients)...")
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
if __name__ == "__main__":
    # 1. Prepare Data
    prepare_data_realistic_non_iid()

    # 2. Run Experiments
    cen_loss = run_centralized()
    fed_loss = run_federated()
    
    # 3. Save Logs to CSV
    max_len = max(len(cen_loss), len(fed_loss))
    cen_loss_padded = cen_loss + [None] * (max_len - len(cen_loss))
    fed_loss_padded = fed_loss + [None] * (max_len - len(fed_loss))
    
    df_logs = pd.DataFrame({
        "Round": range(1, max_len + 1),
        "Centralized_Loss": cen_loss_padded,
        "FedAvg_Loss": fed_loss_padded
    })
    df_logs.to_csv("experiment_4_logs.csv", index=False)
    print("   💾 Saved training logs to 'experiment_4_logs.csv'")
    
    # 4. Plot Graph
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(cen_loss) + 1)
    
    plt.plot(epochs, cen_loss, 'b--', linewidth=2, label='Centralized (Global Benchmark)')
    
    fed_epochs = range(1, len(fed_loss) + 1)
    plt.plot(fed_epochs, fed_loss, 'r-o', linewidth=2, label='FedAvg (Realistic Non-IID)')
    
    plt.title('Performance: Realistic Non-IID (Quantity Skew)', fontsize=14)
    plt.xlabel('Rounds', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig('experiment_4_result.png')
    print(" Graph saved to 'experiment_4_result.png'")
    plt.show()