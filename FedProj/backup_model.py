import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class DrugGNN(torch.nn.Module):
    # ✅ ปรับให้รับค่าจากภายนอกได้เพื่อทำ Tuning
    def __init__(self, hidden_channels=128, dropout_rate=0.072, num_layers=4): 
        super(DrugGNN, self).__init__()
        torch.manual_seed(12345)
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        
        self.node_encoder = torch.nn.Linear(7, hidden_channels)
        
        # สร้าง GCN Layers แบบ Dynamic
        self.convs = torch.nn.ModuleList()
        self.lns = torch.nn.ModuleList()
        
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.lns.append(torch.nn.LayerNorm(hidden_channels))
        
        self.lin1 = torch.nn.Linear(hidden_channels, hidden_channels // 2)
        self.lin2 = torch.nn.Linear(hidden_channels // 2, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = x.float()
        
        x_scaled = x.clone()
        x_scaled[:, 0] = x_scaled[:, 0] / 100.0  
        x_scaled[:, 4] = x_scaled[:, 4] / 100.0  
        
        x = self.node_encoder(x_scaled)
        x = F.leaky_relu(x, negative_slope=0.1)
        
        # ✅ วนลูปผ่าน GCN Layers พร้อมระบบ Skip Connection (ป้องกัน Oversmoothing)
        for i in range(self.num_layers):
            identity = x  # จดจำค่าเดิมไว้
            x = self.convs[i](x, edge_index)
            x = self.lns[i](x)
            x = F.leaky_relu(x, negative_slope=0.1)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            
            # Skip Connection: เอาของใหม่บวกของเก่า (ถ้าไม่ใช่เลเยอร์แรก)
            if i > 0: 
                x = x + identity 
        
        x = global_mean_pool(x, batch)
        
        x = F.leaky_relu(self.lin1(x), negative_slope=0.1)
        x = F.dropout(x, p=self.dropout_rate * 2, training=self.training) # เลเยอร์สุดท้ายเบิ้ล Dropout 
        x = self.lin2(x) 
        
        return x