import pandas as pd
import torch
import numpy as np
from rdkit import Chem
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm
import math 

# ฟังก์ชันแปลง SMILES String -> Graph Data
def molecule_from_smiles(smiles, label):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # 1. Node Features (เหมือนเดิม)
    x = []
    for atom in mol.GetAtoms():
        feature_vector = [
            float(atom.GetAtomicNum()),       
            float(atom.GetFormalCharge()),    
            float(atom.GetDegree()),          
            float(atom.GetIsAromatic()),      
            float(atom.GetMass()),            
            float(atom.GetTotalNumHs()),      
            float(atom.GetHybridization())    
        ]
        x.append(feature_vector)
    x = torch.tensor(x, dtype=torch.float)

    # 2. 🌟 อัปเกรด: Edge Features (พันธะเคมี)
    rows, cols, edge_attrs = [], [], []
    for bond in mol.GetBonds():
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        
        # ดึงคุณสมบัติของพันธะเคมี 3 อย่าง
        edge_feature = [
            float(bond.GetBondTypeAsDouble()), # ชนิดพันธะ (1.0=เดี่ยว, 2.0=คู่, 1.5=อโรมาติก)
            float(bond.GetIsConjugated()),     # เป็น Conjugated system ไหม
            float(bond.IsInRing())             # อยู่ในวงแหวนไหม
        ]
        
        # ใส่ 2 ทิศทาง ไป-กลับ
        rows.extend([start, end])
        cols.extend([end, start])
        edge_attrs.extend([edge_feature, edge_feature])

    if len(rows) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 3), dtype=torch.float) # 3 features
    else:
        edge_index = torch.tensor([rows, cols], dtype=torch.long)
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

    # 3. Label
    label = float(label)
    if label > 100000000: label = 100000000 
    if label < 0.001: label = 0.001        
    molar = label * 1e-9 
    pIC50 = -math.log10(molar)
    y = torch.tensor([[pIC50]], dtype=torch.float)

    # 🌟 ส่ง edge_attr พ่วงไปด้วย
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

# คลาสสำหรับโหลดข้อมูลของแต่ละ Client
class DrugDataset(InMemoryDataset):
    def __init__(self, root, csv_file, transform=None):
        self.csv_file = csv_file
        super(DrugDataset, self).__init__(root, transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.csv_file]

    @property
    def processed_file_names(self):
        # ตั้งชื่อไฟล์ที่แปลงเสร็จแล้ว เช่น 'data_client_diabetes.pt'
        return [f'data_{self.csv_file.replace(".csv", ".pt")}']

    def download(self):
        pass

    def process(self):
        # อ่านไฟล์ CSV
        df = pd.read_csv(self.raw_dir + '/' + self.csv_file)
        print(f"Processing {self.csv_file}...")
        
        data_list = []
        # วนลูปทุกแถวใน CSV เพื่อแปลงเป็น Graph
        for index, row in tqdm(df.iterrows(), total=df.shape[0]):
            smiles = row['canonical_smiles']
            ic50 = row['standard_value'] # ค่าความแรงยา
            
            # แปลงเป็นกราฟ
            graph_data = molecule_from_smiles(smiles, ic50)
            if graph_data is not None:
                data_list.append(graph_data)

        # เซฟเก็บไว้เป็นไฟล์ .pt 
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print(f"✅ Saved {len(data_list)} graphs to {self.processed_paths[0]}")