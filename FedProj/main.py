import flwr as fl
import torch
from torch_geometric.loader import DataLoader
from dataset import DrugDataset
from model import DrugGNN
from client import DrugClient
import matplotlib.pyplot as plt
import os

# --- 1. การตั้งค่า (Configuration) ---
NUM_CLIENTS = 3
NUM_ROUNDS = 10      # จำนวนรอบที่ Server จะสั่งเทรน (เพิ่มได้ถ้าอยากให้แม่นขึ้น)
BATCH_SIZE = 32
FEDPROX_MU = 0.1   # 0.0 = FedAvg, >0.0 = FedProx
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ชื่อไฟล์ที่เราเตรียมไว้ (เรียงตามลำดับ Client ID: 0, 1, 2)
CLIENT_FILES = {
    "0": "client_diabetes.csv",
    "1": "client_heart.csv",
    "2": "client_alzheimers.csv"
}

def load_data(cid):
    """ฟังก์ชันสำหรับโหลดข้อมูลเฉพาะของแต่ละโรงพยาบาล"""
    # แปลง Client ID เป็นชื่อไฟล์
    csv_file = CLIENT_FILES[cid]
    
    # โหลดข้อมูลที่ Process แล้ว (มันจะไปหาไฟล์ .pt ใน folder processed เอง)
    dataset = DrugDataset(root='.', csv_file=csv_file)
    
    # แบ่ง Train 80% / Test 20%
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # สร้าง DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, test_loader

def client_fn(cid: str) -> fl.client.Client:
    """ฟังก์ชันสำหรับสร้าง Client (โรงพยาบาล) เมื่อถึงคิว"""
    print(f"🏥 Starting Client {cid} ({CLIENT_FILES[cid]})...")
    
    # 1. โหลดข้อมูลของตัวเอง
    train_loader, test_loader = load_data(cid)
    
    # 2. สร้างโมเดล
    model = DrugGNN().to(DEVICE)
    
    # 3. ส่งกลับไปให้ Flower (พร้อมตั้งค่า mu สำหรับ FedProx)
    return DrugClient(model, train_loader, test_loader, DEVICE, mu=FEDPROX_MU)

def weighted_average(metrics):
    """ฟังก์ชันคำนวณค่าเฉลี่ยความแม่นยำ (Aggregation)"""
    # metrics เป็น list ของ (num_examples, metric_dict)
    losses = [num_examples * m["mse"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    # Return weighted average loss
    return {"mse": sum(losses) / sum(examples)}

# --- 2. ส่วนเริ่มการทำงาน (Main Simulation) ---
if __name__ == "__main__":
    print(f"Starting Federated Learning Simulation (FedProx mu={FEDPROX_MU})")
    print(f"Using Device: {DEVICE}")

    # กำหนด Strategy (อัลกอริทึมของ Server)
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,  # เลือก Client มาเทรน 100% (ทั้ง 3 แห่ง)
        fraction_evaluate=1.0,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
        evaluate_metrics_aggregation_fn=weighted_average, # วิธีรวมคะแนน
    )

    # เริ่มรัน Simulation!
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0} # ปรับตามสเปคคอม
    )

    # --- 3. แสดงผลกราฟ (Plotting) ---
    print("\n📊 Simulation Finished! Plotting results...")
    
    # ดึงค่า Loss จาก History
    rounds = [data[0] for data in history.losses_distributed]
    loss = [data[1] for data in history.losses_distributed]

    plt.figure(figsize=(10, 6))
    # เติม r ไว้หน้าเครื่องหมายคำพูด
    plt.plot(rounds, loss, marker='o', linestyle='-', color='b', label=rf'FedProx ($\mu$={FEDPROX_MU})')
    plt.title('Federated Learning Training Loss (Drug Discovery)')
    plt.xlabel('Round')
    plt.ylabel('MSE Loss (Lower is Better)')
    plt.grid(True)
    plt.legend()
    plt.savefig('result_chart.png') # เซฟรูปกราฟ
    plt.show()