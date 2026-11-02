import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool # ✅ เปลี่ยนมาใช้ GATv2 และเพิ่ม Max Pool

class DrugGNN(torch.nn.Module):
    # รับค่าจาก Optuna เหมือนเดิม แต่อัปเกรดไส้ใน
    def __init__(self, hidden_channels=128, dropout_rate=0.0519, num_layers=3, heads=2): 
        super(DrugGNN, self).__init__()
        torch.manual_seed(12345)
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        
        # 1. เลเยอร์แปลง 7 Features เบื้องต้น
        self.node_encoder = torch.nn.Linear(7, hidden_channels)
        
        # 2. สร้าง GAT Layers (Graph Attention) แทน GCN
        self.convs = torch.nn.ModuleList()
        self.lns = torch.nn.ModuleList()
        
        for _ in range(num_layers):
            # ใช้ GATv2Conv พร้อมแบ่งหัว Attention เป็น 4 หัว (คล้ายสมอง 4 ซีกช่วยกันมอง)
            # concat=False เพื่อให้ผลลัพธ์สุดท้ายยังคงขนาด hidden_channels เท่าเดิม (128)
            self.convs.append(GATv2Conv(hidden_channels, hidden_channels, heads=heads, concat=False))
            self.lns.append(torch.nn.LayerNorm(hidden_channels))
        
        # 3. ขยายขนาด Linear Layer (เพราะเราจะเอา Mean + Max มาต่อกัน เลยต้องคูณ 2)
        self.lin1 = torch.nn.Linear(hidden_channels * 2, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = x.float()
        
        # Scaling ฟีเจอร์ตามสูตรเดิมของคุณ
        x_scaled = x.clone()
        x_scaled[:, 0] = x_scaled[:, 0] / 100.0  
        x_scaled[:, 4] = x_scaled[:, 4] / 100.0  
        
        x = self.node_encoder(x_scaled)
        x = F.leaky_relu(x, negative_slope=0.1)
        
        # ✅ วนลูปผ่าน GATv2 พร้อม Skip Connection 
        for i in range(self.num_layers):
            identity = x  
            x = self.convs[i](x, edge_index)
            x = self.lns[i](x)
            x = F.leaky_relu(x, negative_slope=0.1)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            
            if i > 0: 
                x = x + identity 
        
        # ✅ Multi-Pooling: ดึงทั้ง "ภาพรวม(Mean)" และ "จุดเด่น(Max)" ของโมเลกุล
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        
        # จับมัดรวมกัน (Concatenate) ขนาดจะกลายเป็น 128 + 128 = 256 มิติ
        x = torch.cat([x_mean, x_max], dim=1)
        
        # Final Prediction เลเยอร์หนาขึ้นนิดนึงเพื่อรับมือข้อมูลที่มากขึ้น
        x = F.leaky_relu(self.lin1(x), negative_slope=0.1)
        x = F.dropout(x, p=self.dropout_rate * 2, training=self.training) 
        x = self.lin2(x) 
        
        return x