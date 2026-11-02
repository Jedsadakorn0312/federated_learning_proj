import optuna
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch.utils.data import ConcatDataset
from model import DrugGNN
from dataset import DrugDataset # อย่าลืม import Dataset มาด้วย

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ✅ รับ train_loader และ val_loader เข้ามาเป็น Argument แทนการโหลดใหม่ทุกรอบ
def objective(trial, train_loader, val_loader):
    # 1. ให้ Optuna สุ่มพารามิเตอร์ที่เราอยากทดสอบ
    hidden_channels = trial.suggest_categorical("hidden_channels", [32, 64, 128])
    num_layers = trial.suggest_int("num_layers", 2, 4)
    dropout_rate = trial.suggest_float("dropout_rate", 0.05, 0.3)
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)

    # 2. สร้างโมเดลตามที่สุ่มได้
    model = DrugGNN(hidden_channels=hidden_channels, 
                    dropout_rate=dropout_rate, 
                    num_layers=num_layers).to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    # 3. ทดลองเทรนสั้นๆ (50 Epochs ก็พอให้เห็นเทรนด์สำหรับ Tuning แล้วครับ)
    epochs = 100
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        for data in train_loader:
            data = data.to(DEVICE)
            optimizer.zero_grad()
            out = model(data)
            target = data.y.view(-1, 1).float() / 10.0
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()

        # ประเมินผลด้วย Validation Set
        model.eval()
        val_loss_total = 0
        steps = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(DEVICE)
                out = model(data)
                target = data.y.view(-1, 1).float() / 10.0
                loss = criterion(out, target)
                val_loss_total += loss.item()
                steps += 1
                
        avg_val_loss = val_loss_total / steps
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

        # แจ้ง Optuna ว่าตอนนี้ Loss เท่าไหร่แล้ว (ช่วยให้มันตัดจบตัวที่แย่ๆ ทิ้งได้)
        trial.report(best_val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val_loss # คืนค่า Loss ที่ดีที่สุดให้ Optuna ไปวิเคราะห์

if __name__ == "__main__":
    print("🚀 Preparing Data for Hyperparameter Tuning...")
    
    # 1. โหลดข้อมูล Train ของทุกโรคมารวมกัน (ทำตัวเป็น Centralized)
    ds_diabetes_train = DrugDataset(root='.', csv_file="train_client_diabetes.csv")
    ds_heart_train = DrugDataset(root='.', csv_file="train_client_heart.csv")
    ds_alz_train = DrugDataset(root='.', csv_file="train_client_alzheimers.csv")
    train_d = ConcatDataset([ds_diabetes_train, ds_heart_train, ds_alz_train])
    
    # 2. โหลดข้อมูล Validation (ชุดสอบเก็บคะแนนระหว่างเทรน)
    val_d = DrugDataset(root='.', csv_file="global_val_set.csv")
    
    # สร้าง DataLoaders ครั้งเดียวจบ
    train_loader = DataLoader(train_d, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_d, batch_size=64, shuffle=False)

    print(f"   📊 Train Size: {len(train_d)} | Validation Size: {len(val_d)}")
    print("🚀 Starting Hyperparameter Tuning with Optuna (100 Trials)...")
    
    # สร้างการทดลอง โดยมีเป้าหมายคือ 'minimize' (ทำ Loss ให้ต่ำที่สุด)
    study = optuna.create_study(direction="minimize")
    
    # สั่งให้ทดลองสุ่ม 100 รูปแบบ
    study.optimize(lambda trial: objective(trial, train_loader, val_loader), n_trials=100)

    print("\n🏆 Tuning Finished!")
    print("Best Trial:")
    trial = study.best_trial
    print(f"  Value (Best Validation Loss): {trial.value:.4f}")
    print("  Best Parameters:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # ==========================================
    # 🌟 NEW: Save Results to CSV
    # ==========================================
    print("\n💾 Saving all tuning results to CSV...")
    df_results = study.trials_dataframe()
    csv_filename = "optuna_tuning_results.csv"
    
    # คุณสามารถเลือกตัดคอลัมน์ที่ไม่จำเป็นออกได้ แต่แนะนำให้เซฟไปทั้งหมดก่อนครับ
    df_results.to_csv(csv_filename, index=False)
    print(f"✅ All 100 trial results successfully saved to '{csv_filename}'!")
    # ==========================================