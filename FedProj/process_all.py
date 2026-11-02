# process_all.py
from dataset import DrugDataset
import os

def process_all_clients():
    # ✅ เปลี่ยนรายชื่อเป็นไฟล์ชุดใหม่ 5 ไฟล์ (Train 3 โรค + Val + Test)
    client_files = [
        'train_client_diabetes.csv',
        'train_client_heart.csv',
        'train_client_alzheimers.csv',
        'global_val_set.csv',
        'global_test_set.csv'
    ]

    print("🚀 เริ่มต้นกระบวนการแปลงข้อมูล (CSV -> Graph)...")
    
    for file_name in client_files:
        # เช็คก่อนว่ามีไฟล์จริงไหม (สมมติว่าคุณเอาไฟล์ทั้ง 5 ไปไว้ในโฟลเดอร์ raw/ แล้ว)
        if not os.path.exists(f'raw/{file_name}'):
            print(f"❌ หาไฟล์ raw/{file_name} ไม่เจอ! ข้ามไป...")
            continue
            
        print(f"\n--- Processing: {file_name} ---")
        try:
            # การเรียกใช้ Class นี้จะกระตุ้นให้มัน process ข้อมูลและเซฟเป็นไฟล์ .pt ทันที
            dataset = DrugDataset(root='.', csv_file=file_name)
            
            # เช็คผลลัพธ์นิดหน่อย
            print(f"✅ เสร็จสมบูรณ์! ได้กราฟทั้งหมด {len(dataset)} กราฟ")
            print(f"   ตัวอย่าง Label (IC50) ของกราฟแรก: {dataset[0].y.item()}")
            
        except Exception as e:
            print(f"⚠️ เกิดข้อผิดพลาดกับไฟล์ {file_name}: {e}")

    print("\n🎉 แปลงไฟล์ทั้งหมดเสร็จสมบูรณ์ พร้อมสำหรับการรัน Experiment แล้ว!")

if __name__ == "__main__":
    process_all_clients()