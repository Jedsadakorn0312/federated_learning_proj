import torch
from torch_geometric.loader import DataLoader
from model import DrugGNN
from dataset import DrugDataset
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

# --- CONFIGURATION ---
BATCH_SIZE = 64
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# รายชื่อโมเดลที่เราจะทดสอบ (Key = ชื่อที่จะโชว์ในกราฟ, Value = ชื่อไฟล์)
MODEL_FILES = {
    "Centralized (Benchmark)": "model_centralized_ex_4.pth",
    "FedAvg (Realistic Non-IID)": "model_fedavg_ex_4.pth", 
}

# ==========================================
# 1. PREPARE TEST DATA (Unseen Data)
# ==========================================
def load_test_data():
    print("🔄 Loading Global Test Data from 'global_test_set.csv'...")
    if not os.path.exists("global_test_set.csv"):
        raise FileNotFoundError("❌ ไม่เจอไฟล์ 'global_test_set.csv'! กรุณารัน split_csv.py ก่อนครับ")

    test_d = DrugDataset(root='.', csv_file="global_test_set.csv")
    print(f"   📊 Test Samples (Unseen Data): {len(test_d)}")
    return DataLoader(test_d, batch_size=BATCH_SIZE, shuffle=False)

# ==========================================
# 2. EVALUATION FUNCTION
# ==========================================
def evaluate_model(model_path, test_loader):
    print(f"\n🧪 Evaluating: {model_path} ...")
    if not os.path.exists(model_path):
        print(f"   ⚠️ Warning: File '{model_path}' not found. Skipping.")
        return None

    model = DrugGNN().to(DEVICE)
    try:
        state_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        return None

    model.eval()
    
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for data in test_loader:
            data = data.to(DEVICE)
            out = model(data)
            
            # ✅ แก้ไขตรงนี้: เติม .flatten() ให้ y_true เพื่อให้เป็นตัวเลขล้วนๆ
            y_true.extend(data.y.cpu().numpy().flatten())
            
            prediction = out.cpu().numpy().flatten() * 10.0
            y_pred.extend(prediction)
            
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"   ✅ MSE: {mse:.4f}")
    print(f"   ✅ R2 Score: {r2:.4f}")
    
    return {
        "mse": mse, 
        "mae": mae, 
        "r2": r2, 
        "y_true": np.array(y_true), # แปลงเป็น numpy array เพื่อความชัวร์
        "y_pred": np.array(y_pred)
    }

# ==========================================
# 3. MAIN SCRIPT
# ==========================================
if __name__ == "__main__":
    test_loader = load_test_data()
    results = {}
    
    for name, filepath in MODEL_FILES.items():
        res = evaluate_model(filepath, test_loader)
        if res is not None:
            results[name] = res

    if results:
        sorted_results = sorted(results.items(), key=lambda x: x[1]["mse"])
        names = [item[0] for item in sorted_results]
        mse_scores = [item[1]["mse"] for item in sorted_results]
        r2_scores = [item[1]["r2"] for item in sorted_results]
        
        # 📊 Plot MSE
        fig, ax = plt.subplots(figsize=(10, 5))
        # ✅ แก้ไข Warning: ใส่ hue=names และ legend=False
        sns.barplot(x=names, y=mse_scores, hue=names, legend=False, palette="viridis", ax=ax)
        ax.set_title("Model Comparison: MSE Loss (Lower is Better)")
        ax.set_ylabel("Mean Squared Error")
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        for i, v in enumerate(mse_scores):
            ax.text(i, v + 0.01, f"{v:.4f}", ha='center', va='bottom', fontweight='bold')
        plt.savefig("evaluation_mse_comparison.png")
        plt.clf()
        print("\n✅ Saved 'evaluation_mse_comparison.png'")
        
        # 📊 Plot R2 Score
        fig, ax = plt.subplots(figsize=(10, 5))
        # ✅ แก้ไข Warning: ใส่ hue=names และ legend=False
        sns.barplot(x=names, y=r2_scores, hue=names, legend=False, palette="magma", ax=ax)
        ax.set_title("Model Comparison: R2 Score (Higher is Better)")
        ax.set_ylabel("R2 Score")
        ax.set_ylim(0, 1) 
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        for i, v in enumerate(r2_scores):
            ax.text(i, v + 0.01, f"{v:.4f}", ha='center', va='bottom', fontweight='bold')
        plt.savefig("evaluation_r2_comparison.png")
        plt.clf()
        print("✅ Saved 'evaluation_r2_comparison.png'")

        # 🎯 Plot Scatter
        best_model = names[0] 
        data = results[best_model]
        
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(data["y_true"], data["y_pred"], alpha=0.5, color='blue')
        
        # ✅ ใช้ np.min / np.max ชัวร์กว่าในการหาจุดต่ำสุดสูงสุด
        min_val = min(np.min(data["y_true"]), np.min(data["y_pred"]))
        max_val = max(np.max(data["y_true"]), np.max(data["y_pred"]))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        
        ax.set_title(f"Prediction Accuracy: {best_model}")
        ax.set_xlabel("Actual pIC50")
        ax.set_ylabel("Predicted pIC50")
        ax.grid(True)
        plt.savefig("evaluation_scatter_plot.png")
        plt.clf()
        print("✅ Saved 'evaluation_scatter_plot.png'")

    else:
        print("❌ No models were evaluated.")