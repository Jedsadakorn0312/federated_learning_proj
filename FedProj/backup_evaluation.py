import torch
from torch_geometric.loader import DataLoader
from model import DrugGNN
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
    "FedAvg Realistic NonIID": "model_centralized_ex_2.pth",
    # ถ้ามีไฟล์อื่นเพิ่ม เช่น FedProx ก็ใส่เพิ่มตรงนี้ได้
    # "FedProx (mu=0.001)": "model_fedprox_mu0.001.pth",
}

# ==========================================
# 1. PREPARE TEST DATA
# ==========================================
def load_test_data():
    print("🔄 Loading Test Data from 'centralized_data.pt'...")
    
    if not os.path.exists("centralized_data_ex4.pt"):
        raise FileNotFoundError("❌ ไม่เจอไฟล์ 'centralized_data.pt'! กรุณารัน experiment_3.py ก่อนครับ")

    # โหลดไฟล์ (เราสนใจแค่ test_d)
    _, test_d = torch.load("centralized_data.pt")
    print(f"   📊 Test Samples: {len(test_d)}")
    
    return DataLoader(test_d, batch_size=BATCH_SIZE, shuffle=False)

# ==========================================
# 2. EVALUATION FUNCTION
# ==========================================
def evaluate_model(model_path, test_loader):
    print(f"\n🧪 Evaluating: {model_path} ...")
    
    if not os.path.exists(model_path):
        print(f"   ⚠️ Warning: File '{model_path}' not found. Skipping.")
        return None

    # Load Model Structure
    model = DrugGNN().to(DEVICE)
    
    # Load Weights
    try:
        state_dict = torch.load(model_path)
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
            
            y_true.extend(data.y.cpu().numpy())
            prediction = out.cpu().numpy().flatten() * 10.0
            y_pred.extend(prediction)
            
    # Calculate Metrics
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"   ✅ MSE: {mse:.4f}")
    print(f"   ✅ R2 Score: {r2:.4f}")
    
    return {
        "mse": mse, 
        "mae": mae, 
        "r2": r2, 
        "y_true": y_true, 
        "y_pred": y_pred
    }

# ==========================================
# 3. MAIN SCRIPT
# ==========================================
if __name__ == "__main__":
    # 1. Load Data
    test_loader = load_test_data()
    
    results = {}
    
    # 2. Evaluate All Models
    for name, filepath in MODEL_FILES.items():
        res = evaluate_model(filepath, test_loader)
        if res is not None:
            results[name] = res

    # 3. PLOT BAR CHART (Comparison)
    if results:
        names = list(results.keys())
        mse_scores = [results[n]["mse"] for n in names]
        r2_scores = [results[n]["r2"] for n in names]
        
        # Plot MSE
        plt.figure(figsize=(10, 5))
        sns.barplot(x=names, y=mse_scores, palette="viridis")
        plt.title("Model Comparison: MSE Loss (Lower is Better)")
        plt.ylabel("Mean Squared Error")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        for i, v in enumerate(mse_scores):
            plt.text(i, v + 0.01, f"{v:.4f}", ha='center', va='bottom', fontweight='bold')
        plt.savefig("evaluation_mse_comparison.png")
        print("\n✅ Saved 'evaluation_mse_comparison.png'")
        
        # Plot R2 Score
        plt.figure(figsize=(10, 5))
        sns.barplot(x=names, y=r2_scores, palette="magma")
        plt.title("Model Comparison: R2 Score (Higher is Better)")
        plt.ylabel("R2 Score")
        plt.ylim(0, 1) # R2 usually between 0 and 1
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        for i, v in enumerate(r2_scores):
            plt.text(i, v + 0.01, f"{v:.4f}", ha='center', va='bottom', fontweight='bold')
        plt.savefig("evaluation_r2_comparison.png")
        print("✅ Saved 'evaluation_r2_comparison.png'")

        # 4. PLOT SCATTER (Prediction vs True) - Optional but Cool for Thesis
        # Plotting the best model (usually Centralized)
        best_model = names[0] # Assume first one is best/baseline
        data = results[best_model]
        
        plt.figure(figsize=(6, 6))
        plt.scatter(data["y_true"], data["y_pred"], alpha=0.5, color='blue')
        
        # Draw perfect line
        min_val = min(min(data["y_true"]), min(data["y_pred"]))
        max_val = max(max(data["y_true"]), max(data["y_pred"]))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        
        plt.title(f"Prediction Accuracy: {best_model}")
        plt.xlabel("Actual pIC50")
        plt.ylabel("Predicted pIC50")
        plt.grid(True)
        plt.savefig("evaluation_scatter_plot.png")
        print("✅ Saved 'evaluation_scatter_plot.png'")

        plt.show()
    else:
        print("❌ No models were evaluated.")