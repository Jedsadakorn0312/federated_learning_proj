from chembl_webresource_client.new_client import new_client
import pandas as pd

activity = new_client.activity

def get_data_by_id(target_id, disease_label, target_count=5000):
    print(f"\nกำลังดึงข้อมูลสำหรับ {disease_label} (ID: {target_id})...")
    
    # Filter เฉพาะค่า IC50 ในหน่วย nM
    res = activity.filter(target_chembl_id=target_id)\
                  .filter(standard_type="IC50")\
                  .filter(standard_units="nM")
    
    df = pd.DataFrame.from_dict(res)
    
    if df.empty:
        print(f"❌ ไม่พบข้อมูลสำหรับ {disease_label}")
        return None
        
    # Clean Data
    df = df[['canonical_smiles', 'standard_value', 'standard_units']]
    df = df.dropna()
    
    # แปลงค่าตัวเลขและลบค่าที่ไม่ใช่ตัวเลขออก
    df['standard_value'] = pd.to_numeric(df['standard_value'], errors='coerce')
    df = df.dropna()
    
    # ลบยาซ้ำ (Keep unique compounds)
    df = df.drop_duplicates(subset=['canonical_smiles'])
    
    # --- คุมจำนวนข้อมูล (Data Cap) ---
    if len(df) > target_count:
        df = df.sample(n=target_count, random_state=42)
        print(f"   (ตัดข้อมูลเหลือ {target_count} รายการ เพื่อความสมดุล)")
    
    df['disease'] = disease_label
    print(f"✅ สำเร็จ: ได้ข้อมูล {len(df)} รายการ สำหรับ {disease_label}")
    return df

# --- THE AGING SOCIETY CLUSTER ---

# 1. Diabetes (เบาหวาน): Target DPP-4
# ID: CHEMBL284 -> ข้อมูลเยอะและเสถียร
df_diabetes = get_data_by_id("CHEMBL284", "Diabetes", target_count=4000)

# 2. Heart Disease (ความดัน/หัวใจ): Target Angiotensin II Receptor Type 1
# ID: CHEMBL227 -> ข้อมูลเยอะกว่า ACE (ตัวเก่า)
df_heart = get_data_by_id("CHEMBL227", "Heart_Disease", target_count=4000)

# 3. Alzheimer's Disease (อัลไซเมอร์): Target Acetylcholinesterase
# ID: CHEMBL220 -> ข้อมูลมหาศาล และเป็นตัวแทนกลุ่มยา CNS (สมอง)
df_alzheimer = get_data_by_id("CHEMBL220", "Alzheimers", target_count=4000)

# --- Save Files ---
print("\n--- Saving Files ---")
if df_diabetes is not None: 
    df_diabetes.to_csv("client_diabetes.csv", index=False)
    print("Saved client_diabetes.csv")
    
if df_heart is not None: 
    df_heart.to_csv("client_heart.csv", index=False)
    print("Saved client_heart.csv")
    
if df_alzheimer is not None: 
    df_alzheimer.to_csv("client_alzheimers.csv", index=False)
    print("Saved client_alzheimers.csv")