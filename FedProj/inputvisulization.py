from rdkit import Chem
from rdkit.Chem import Draw
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 1. โหลดข้อมูลโมเลกุลจาก SMILES
smiles = "[NH3+][C@@H](Cc1ccccc1)C(=O)[O-]"
mol = Chem.MolFromSmiles(smiles)

print(f"🧪 กำลังประมวลผล SMILES: {smiles}")

# 2. วาดรูป 2D ของสารเคมี (เหมือนที่นักเคมีดู)
Draw.MolToFile(mol, 'molecule_2d.png', size=(600, 400))
print("✅ สร้างรูปสารเคมี 2D สำเร็จ เซฟเป็น -> 'molecule_2d.png'")

# ==========================================
# 3. สกัด 7 ฟีเจอร์ของแต่ละ Node (data.x)
# ==========================================
features = []
for atom in mol.GetAtoms():
    feat = [
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        int(atom.GetHybridization()), 
        int(atom.GetIsAromatic()),
        atom.GetImplicitValence(),
        int(atom.GetChiralTag())
    ]
    features.append(feat)

df_features = pd.DataFrame(features, columns=['AtomicNum', 'Degree', 'Charge', 'Hybridization', 'IsAromatic', 'ImplicitHs', 'Chirality'])
df_features.index.name = 'NodeIdx'
df_features.to_csv('node_features.csv')
print(f"✅ สกัด Node Features สำเร็จ เซฟเป็น -> 'node_features.csv'")
print(f"   📦 รูปร่าง Matrix ของ data.x: {df_features.shape} (โหนด x 7 ฟีเจอร์)")

# ==========================================
# 4. สกัดเส้นเชื่อม (data.edge_index)
# ==========================================
src = []
dst = []
for bond in mol.GetBonds():
    u = bond.GetBeginAtomIdx()
    v = bond.GetEndAtomIdx()
    
    # กราฟของ GNN เป็นแบบไร้ทิศทาง (Undirected) ต้องเพิ่มเส้นทางแบบไป-กลับ
    src.extend([u, v])
    dst.extend([v, u])

edge_index = np.array([src, dst])
df_edges = pd.DataFrame(edge_index.T, columns=['Source', 'Target'])
df_edges.to_csv('edge_index.csv', index=False)
print(f"✅ สกัด Edge Index สำเร็จ เซฟเป็น -> 'edge_index.csv'")
print(f"   📦 รูปร่าง Matrix ของ data.edge_index: {edge_index.shape} (2 x เส้นเชื่อมทั้งหมด)")

# ==========================================
# 5. วาดกราฟโครงข่ายคณิตศาสตร์ด้วย NetworkX
# ==========================================
G = nx.Graph()

# เพิ่มโหนดพร้อมใส่ชื่อกำกับ
for i in range(mol.GetNumAtoms()):
    symbol = mol.GetAtomWithIdx(i).GetSymbol()
    G.add_node(i, label=f"{symbol}{i}") # เช่น O0, C1, N11

# เพิ่มเส้นเชื่อม
for bond in mol.GetBonds():
    G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

plt.figure(figsize=(10, 8))
pos = nx.kamada_kawai_layout(G) # จัดเรียงโหนดให้สวยงามตามแรงดึงดูด
labels = nx.get_node_attributes(G, 'label')

# กำหนดสีให้โหนด (O=แดง, N=ฟ้า, C=เทา) เพื่อให้อ่านง่าย
colors = []
for n in G.nodes():
    sym = mol.GetAtomWithIdx(n).GetSymbol()
    if sym == 'O': 
        colors.append('#ff9999')
    elif sym == 'N': 
        colors.append('#99ccff')
    else: 
        colors.append('#cccccc')

nx.draw(G, pos, with_labels=True, labels=labels, node_color=colors, 
        node_size=800, font_size=10, font_weight='bold', edge_color='#555555')

plt.title("Graph Structure for GNN (Nodes & Edges)", fontsize=16)
plt.savefig('molecule_graph.png', bbox_inches='tight', dpi=150)
print("✅ สร้างรูปโครงข่ายกราฟสำเร็จ เซฟเป็น -> 'molecule_graph.png'")

plt.show()