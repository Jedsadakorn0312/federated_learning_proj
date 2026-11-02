import pandas as pd
import os

# โฟลเดอร์ที่เก็บไฟล์ต้นฉบับ
INPUT_DIR = "raw"

# ชื่อไฟล์ต้นฉบับ
FILES = ["client_diabetes.csv", "client_heart.csv", "client_alzheimers.csv"]

print(f"🚀 Starting CSV Data Splitting (70% Train / 10% Validation / 20% Global Test)...")

global_test_dfs = []
global_val_dfs = [] 

for filename in FILES:
    file_path = os.path.join(INPUT_DIR, filename)
    
    if not os.path.exists(file_path):
        print(f"⚠️ Warning: File '{file_path}' not found! Please check the folder.")
        continue
        
    # 1. อ่านไฟล์ CSV จากโฟลเดอร์ raw
    df = pd.read_csv(file_path)
    
    # 2. สับไพ่ก่อนแบ่ง เพื่อให้ข้อมูลกระจายตัวแบบสุ่มจริงๆ
    df_shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    # 3. คำนวณจำนวนแถวสำหรับ 70 / 10 / 20
    total_len = len(df_shuffled)
    train_len = int(0.7 * total_len)
    val_len = int(0.1 * total_len)
    
    # 4. ใช้การหั่น Index (Slicing) เพื่อแบ่งกอง ป้องกันข้อมูลปนกัน 100%
    train_df = df_shuffled.iloc[:train_len]
    val_df = df_shuffled.iloc[train_len : train_len + val_len]
    test_df = df_shuffled.iloc[train_len + val_len:]
    
    # 5. เซฟไฟล์ส่วน Train (70%) สำหรับให้ Client เอาไปเรียนรู้
    train_filename = f"train_{filename}"
    train_df.to_csv(train_filename, index=False)
    print(f"   ✅ Saved {train_filename} ({len(train_df)} rows)")
    
    # 6. เก็บไฟล์ Val (10%) และ Test (20%) ไปรวมร่างเป็น Global
    global_val_dfs.append(val_df)
    global_test_dfs.append(test_df)

# --- สร้าง Global Validation Set (10%) ---
if global_val_dfs:
    global_val_df = pd.concat(global_val_dfs, ignore_index=True)
    # สับไพ่ผสมโรค
    global_val_df = global_val_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    global_val_df.to_csv("global_val_set.csv", index=False)
    print(f"\n📊 Saved 'global_val_set.csv' ({len(global_val_df)} rows) -> ใช้ตัดสินใจระหว่างเทรน (LR Scheduler / Save Best Model)")

# --- สร้าง Global Test Set (20%) ---
if global_test_dfs:
    global_test_df = pd.concat(global_test_dfs, ignore_index=True)
    # สับไพ่ผสมโรค
    global_test_df = global_test_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    global_test_df.to_csv("global_test_set.csv", index=False)
    print(f"🌍 Saved 'global_test_set.csv' ({len(global_test_df)} rows) -> 🔒 ล็อกกุญแจเก็บไว้! ใช้สร้างกราฟตอนจบโปรเจกต์เท่านั้น!")