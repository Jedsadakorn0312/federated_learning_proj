import torch
from torch_geometric.loader import DataLoader
import flwr as fl
from collections import OrderedDict
import copy

# นำเข้า Model และ Dataset ที่เราทำไว้
from model import DrugGNN
from dataset import DrugDataset

# ฟังก์ชันสำหรับเทรน (Training Loop)
# รองรับ FedProx ด้วยพารามิเตอร์ mu
def train(model, train_loader, epochs, lr, device, mu=0.0, global_params=None):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1.84e-05)
    
    # ✅ FIX 1: เพิ่ม Scheduler ให้ Client ด้วย (ใช้ patience=1 เพราะเทรนแค่ 3 Epoch)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        steps = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            
            out = model(data)
            
            # ✅ FIX 2: เติม .0 เพื่อความชัวร์ของ Type (Float)
            target = data.y.view(-1, 1).float() / 10.0
            loss = criterion(out, target)

            # ส่วนของ FedProx (ถ้ามี mu > 0)
            if mu > 0.0 and global_params is not None:
                proximal_term = 0.0
                for local_weights, global_weights in zip(model.parameters(), global_params):
                    proximal_term += (local_weights - global_weights).norm(2)
                
                loss += (mu / 2) * proximal_term

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            steps += 1
            
        # อัปเดต Scheduler ตอนจบแต่ละ Epoch ใน Client
        avg_train_loss = total_loss / steps
        # scheduler.step(avg_train_loss)

# ฟังก์ชันสำหรับทดสอบ (Test Loop)
def test(model, test_loader, device):
    criterion = torch.nn.MSELoss()
    model.eval()
    
    total_loss = 0.0
    steps = 0
    
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data)
            
            # ✅ เติม .0 ให้ตรงกัน
            target = data.y.view(-1, 1).float() / 10.0
            loss = criterion(out, target)
            
            total_loss += loss.item()
            steps += 1

    if steps == 0:
        return 0.0, 0.0

    # ✅ FIX 3: ลบ return ที่ซ้ำซ้อนออก
    return total_loss / steps, 0.0

class DrugClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader, device, mu=0.0):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.mu = mu # ค่าความเข้มข้นของ FedProx (0=ปิด, >0=เปิด)

    def get_parameters(self, config):
        # ส่งค่า Weight ปัจจุบันกลับไป Server
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        # รับค่า Weight จาก Server มาอัปเดตใส่ตัวเอง
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        global_params = [p.clone().detach() for p in self.model.parameters()]

        train(self.model, self.train_loader, epochs=3, lr=0.000416, 
              device=self.device, mu=self.mu, global_params=global_params)

        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        
        # Unpack the tuple returned by test()
        loss, accuracy = test(self.model, self.test_loader, self.device)
        
        # Return loss, number of examples, and a dictionary of metrics
        return float(loss), len(self.test_loader.dataset), {"mse": float(loss)}