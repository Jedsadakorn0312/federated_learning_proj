# import pandas as pd
# import torch
# import numpy as np
# from rdkit import Chem
# from torch_geometric.data import Data, InMemoryDataset
# from tqdm import tqdm
# import math 

# # ฟังก์ชันแปลง SMILES String -> Graph Data
# def molecule_from_smiles(smiles, label):
#     # 1. แปลงตัวหนังสือเป็นโมเลกุลด้วย RDKit
#     mol = Chem.MolFromSmiles(smiles)
#     if mol is None:
#         return None

#     # 2. 🧬 อัปเกรด: ดึงคุณสมบัติของอะตอม (Nodes) แบบจัดเต็ม 7 ฟีเจอร์!
#     x = []
#     for atom in mol.GetAtoms():
#         feature_vector = [
#             float(atom.GetAtomicNum()),       # 1. ประเภทอะตอม (Atomic Number)
#             float(atom.GetFormalCharge()),    # 2. ประจุไฟฟ้า (Formal Charge)
#             float(atom.GetDegree()),          # 3. จำนวนพันธะ (Degree)
#             float(atom.GetIsAromatic()),      # 4. อโรมาติก (1.0 = เป็น, 0.0 = ไม่เป็น)
#             float(atom.GetMass()),            # 5. มวลอะตอม (Atomic Mass)
#             float(atom.GetTotalNumHs()),      # 6. จำนวนไฮโดรเจนที่เกาะอยู่ (Total Hydrogens)
#             float(atom.GetHybridization())    # 7. การจัดเรียงอิเล็กตรอน (Hybridization state)
#         ]
#         x.append(feature_vector)
    
#     # ตอนนี้ x จะมีขนาด [จำนวนอะตอม, 7] แทนที่จะเป็น [จำนวนอะตอม, 1] แบบเมื่อก่อน
#     x = torch.tensor(x, dtype=torch.float)

#     # 3. ดึงข้อมูลพันธะเคมี (Edges)
#     rows, cols = [], []
#     for bond in mol.GetBonds():
#         start = bond.GetBeginAtomIdx()
#         end = bond.GetEndAtomIdx()
#         # กราฟเคมีเป็นแบบ "ไป-กลับ" (Undirected) ต้องใส่ทั้ง 2 ทิศทาง
#         rows.extend([start, end])
#         cols.extend([end, start])

#     # เช็คเผื่อโมเลกุลมีแค่อะตอมเดียวไม่มีพันธะ
#     if len(rows) == 0:
#         edge_index = torch.empty((2, 0), dtype=torch.long)
#     else:
#         edge_index = torch.tensor([rows, cols], dtype=torch.long)

#     # 4. จัดการ Label (pIC50)
#     label = float(label)
#     if label > 100000000: label = 100000000 # กันค่าระเบิด
#     if label < 0.001: label = 0.001        # กันค่าเป็น 0 เดี๋ยว log error
    
#     # แปลงหน่วย nM เป็น Molar แล้ว take log
#     molar = label * 1e-9 
#     pIC50 = -math.log10(molar)
    
#     # เก็บค่า pIC50 แทนค่าเดิม
#     y = torch.tensor([[pIC50]], dtype=torch.float)

#     return Data(x=x, edge_index=edge_index, y=y)

# # คลาสสำหรับโหลดข้อมูลของแต่ละ Client
# class DrugDataset(InMemoryDataset):
#     def __init__(self, root, csv_file, transform=None):
#         self.csv_file = csv_file
#         super(DrugDataset, self).__init__(root, transform)
#         self.data, self.slices = torch.load(self.processed_paths[0])

#     @property
#     def raw_file_names(self):
#         return [self.csv_file]

#     @property
#     def processed_file_names(self):
#         # ตั้งชื่อไฟล์ที่แปลงเสร็จแล้ว เช่น 'data_client_diabetes.pt'
#         return [f'data_{self.csv_file.replace(".csv", ".pt")}']

#     def download(self):
#         pass

#     def process(self):
#         # อ่านไฟล์ CSV
#         df = pd.read_csv(self.raw_dir + '/' + self.csv_file)
#         print(f"Processing {self.csv_file}...")
        
#         data_list = []
#         # วนลูปทุกแถวใน CSV เพื่อแปลงเป็น Graph
#         for index, row in tqdm(df.iterrows(), total=df.shape[0]):
#             smiles = row['canonical_smiles']
#             ic50 = row['standard_value'] # ค่าความแรงยา
            
#             # แปลงเป็นกราฟ
#             graph_data = molecule_from_smiles(smiles, ic50)
#             if graph_data is not None:
#                 data_list.append(graph_data)

#         # เซฟเก็บไว้เป็นไฟล์ .pt 
#         data, slices = self.collate(data_list)
#         torch.save((data, slices), self.processed_paths[0])
#         print(f"✅ Saved {len(data_list)} graphs to {self.processed_paths[0]}")