import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import matplotlib.pyplot as plt
import networkx as nx # เพิ่มสำหรับการวาดแผนภาพโครงสร้าง
from chembl_webresource_client.new_client import new_client
from rdkit import Chem
from rdkit.Chem import AllChem, rdFingerprintGenerator

# ==========================================
# PART 1: Data Fetching (ดึงข้อมูล)
# ==========================================
print("--- 1. Data Fetching ---")
def get_chembl_data(target_id, disease_label, limit=1000):
    activity = new_client.activity
    res = activity.filter(target_chembl_id=target_id) \
                  .filter(standard_type="IC50") \
                  .filter(standard_relation="=") \
                  .filter(standard_units="nM")
    
    data = []
    count = 0
    for item in res:
        if count >= limit: break
        if item['canonical_smiles'] and item['standard_value']:
            try:
                val = float(item['standard_value'])
                data.append({
                    'smiles': item['canonical_smiles'],
                    'IC50_nM': val,
                    'disease': disease_label
                })
                count += 1
            except:
                continue
    print(f"✅ {disease_label}: {len(data)} rows")
    return pd.DataFrame(data)

# ดึงข้อมูลจริง (ลด limit ลงเล็กน้อยเพื่อความเร็วในการ demo สามารถเพิ่มได้)
df_alz = get_chembl_data('CHEMBL220', 'Alzheimer', limit=1500)
df_dia = get_chembl_data('CHEMBL284', 'Diabetes', limit=1500)
df_hrt = get_chembl_data('CHEMBL1806', 'Heart', limit=1000)

df_all = pd.concat([df_alz, df_dia, df_hrt], ignore_index=True)
df_all['pIC50'] = -np.log10(df_all['IC50_nM'] * 1e-9)
print(f"Total Data: {len(df_all)} rows")

# ==========================================
# PART 2: Feature Engineering (SMILES -> Vector)
# ==========================================
print("\n--- 2. Feature Engineering ---")
INPUT_SIZE = 1024 
mfgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=INPUT_SIZE)

def smiles_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return np.zeros((INPUT_SIZE,))
    return mfgen.GetFingerprintAsNumPy(mol)

X_data = np.array([smiles_to_fp(s) for s in df_all['smiles']])
y_data = df_all['pIC50'].values.reshape(-1, 1)

X_tensor = torch.tensor(X_data, dtype=torch.float32)
y_tensor = torch.tensor(y_data, dtype=torch.float32)

# ==========================================
# PART 3: Data Distribution (จัดสรรลงโรงพยาบาล)
# ==========================================
print("\n--- 3. Data Distribution ---")

# แยก index ตามโรค
idx_alz = df_all[df_all['disease'] == 'Alzheimer'].index.to_numpy()
idx_dia = df_all[df_all['disease'] == 'Diabetes'].index.to_numpy()
idx_hrt = df_all[df_all['disease'] == 'Heart'].index.to_numpy()

# แบ่งข้อมูล Training ให้ A, B, C, D (ใช้ 90% ของแต่ละโรค)
def get_train_indices(indices, ratio=0.9):
    np.random.shuffle(indices)
    split = int(len(indices) * ratio)
    return indices[:split]

train_alz = get_train_indices(idx_alz)
train_dia = get_train_indices(idx_dia)
train_hrt = get_train_indices(idx_hrt)

# จัดลงถัง (Clients)
mid_dia = len(train_dia) // 2
data_A = (X_tensor[train_dia[:mid_dia]], y_tensor[train_dia[:mid_dia]]) # Diabetes 1
data_B = (X_tensor[train_dia[mid_dia:]], y_tensor[train_dia[mid_dia:]]) # Diabetes 2
data_C = (X_tensor[train_hrt], y_tensor[train_hrt])                       # Heart
data_D = (X_tensor[train_alz], y_tensor[train_alz])                       # Alzheimer

# --- [แก้ไข] Hospital E รวมข้อมูลทั้งหมด (All Data) ---
# เพื่อใช้เป็น Global Benchmark (ทั้ง Train + Test ของทุกโรค)
data_E_X = X_tensor 
data_E_y = y_tensor

clients = {
    "Hosp A (Diabetes)": data_A,
    "Hosp B (Diabetes)": data_B,
    "Hosp C (Heart)": data_C,
    "Hosp D (Alzheimer)": data_D
}

print(f"Client Data Sizes:")
for k, v in clients.items():
    print(f" - {k}: {len(v[0])}")
print(f" - Hosp E (ALL DATA): {len(data_E_X)} (Used for Final Eval)")

# ==========================================
# PART 4: Model & FL Setup
# ==========================================
class DrugModel(nn.Module):
    def __init__(self):
        super(DrugModel, self).__init__()
        self.layer1 = nn.Linear(INPUT_SIZE, 256) # ขยาย model หน่อย
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 1)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        return x

def train_local(model, data, epochs=3, lr=0.001):
    X, y = data
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    total_loss = 0
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return model.state_dict(), total_loss / epochs

# ==========================================
# PART 5: FL Simulation Loop
# ==========================================
print("\n--- 4. Federated Learning Simulation ---")
global_model = DrugModel()
rounds = 10
loss_history = {name: [] for name in clients}
global_loss_history = [] # เก็บผล E ด้วย

for r in range(rounds):
    local_weights = []
    global_dict = global_model.state_dict()
    print(f"Round {r+1}/{rounds}")
    
    # 1. Local Training
    for name, data in clients.items():
        local_net = DrugModel()
        local_net.load_state_dict(copy.deepcopy(global_dict))
        w, loss = train_local(local_net, data)
        local_weights.append(copy.deepcopy(w))
        loss_history[name].append(loss)
        # print(f"  > {name} Loss: {loss:.4f}") # ปิด print ย่อยเพื่อความสะอาด

    # 2. Aggregation (FedAvg)
    avg_weights = copy.deepcopy(local_weights[0])
    for key in avg_weights.keys():
        for i in range(1, len(local_weights)):
            avg_weights[key] += local_weights[i][key]
        avg_weights[key] = torch.div(avg_weights[key], len(local_weights))
    global_model.load_state_dict(avg_weights)

    # 3. Global Evaluation on Hospital E (All Data)
    global_model.eval()
    with torch.no_grad():
        pred_E = global_model(data_E_X)
        loss_E = nn.MSELoss()(pred_E, data_E_y).item()
        global_loss_history.append(loss_E)
        print(f"  >>> Global Eval (Hosp E - All Data) Loss: {loss_E:.4f}")

# ==========================================
# PART 6: Visualizations (Graph & Diagram)
# ==========================================

# 6.1 Learning Curve Graph
plt.figure(figsize=(10, 6))
markers = ['o', 's', '^', 'D']
for i, (name, losses) in enumerate(loss_history.items()):
    plt.plot(range(1, rounds+1), losses, marker=markers[i], label=name, alpha=0.7)

# พล็อตเส้น Global E ให้หนาและเด่น
plt.plot(range(1, rounds+1), global_loss_history, 'r--', linewidth=3, label='Global Model (Hosp E - All Data)')

plt.title('Federated Learning Convergence', fontsize=14)
plt.xlabel('Round')
plt.ylabel('MSE Loss')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('fl_learning_curve.png')
print("\n✅ Graph saved: 'fl_learning_curve.png'")

# 6.2 Architecture Diagram (แผนภาพจำลอง)
def draw_architecture():
    G = nx.DiGraph()
    server = "Central Server\n(Aggregator)"
    hosp_E = "Hospital E\n(Global Eval\nALL DATA)"
    
    # Nodes
    G.add_node(server, pos=(0.5, 0.9), color='lightgreen')
    G.add_node(hosp_E, pos=(0.9, 0.5), color='salmon') # E อยู่แยกออกมา
    
    client_nodes = list(clients.keys())
    for i, c in enumerate(client_nodes):
        G.add_node(c, pos=(0.2, 0.8 - i*0.2), color='skyblue')
        G.add_edge(server, c, label="Global Model")
        G.add_edge(c, server, label="Update")
    
    G.add_edge(server, hosp_E, label="Validate")

    pos = nx.get_node_attributes(G, 'pos')
    colors = nx.get_node_attributes(G, 'color').values()
    
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color=colors, node_size=3500, 
            font_size=8, font_weight='bold', node_shape='s', 
            arrows=True, arrowsize=20, edge_color='gray')
    
    # Draw Edge Labels
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
    plt.title("Federated Learning System Architecture", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('fl_architecture_diagram.png')
    print("✅ Diagram saved: 'fl_architecture_diagram.png'")

draw_architecture()
plt.show() # แสดงผลกราฟ