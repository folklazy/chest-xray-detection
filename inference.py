import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# Import โมเดลของเรา (ต้องชื่อตรงกับไฟล์ src/model.py)
from src.model import CheXpertLightning

# ==========================================
# ⚙️ CONFIG (แก้ตรงนี้ก่อนรัน!)
# ==========================================

# 1. Path ของไฟล์ Model ที่ดีที่สุด (ก๊อปมาจาก Log ตอนเทรนเสร็จ)
# ตัวอย่าง: "logs/chexpert/version_2/checkpoints/best-epoch=04-val_auc=0.8023.ckpt"
CKPT_PATH = r"logs/chexpert/version_2/checkpoints/best-epoch=04-val_auc=0.8023.ckpt" 

# 2. Path ของรูป X-ray ที่จะทดสอบ
# เลือกรูปใน folder valid มาลองสักรูป
IMG_PATH = r"data/CheXpert-v1.0-small/valid/patient64541/study1/view1_frontal.jpg"

# 3. ตั้งค่า Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 🛠️ HELPER FUNCTIONS
# ==========================================

def load_model(ckpt_path):
    print(f"🔄 Loading model from: {ckpt_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"❌ ไม่เจอไฟล์ Model ที่: {ckpt_path}")
    
    # โหลด Lightning Module
    model = CheXpertLightning.load_from_checkpoint(ckpt_path)
    model.to(DEVICE)
    model.eval() # ปิด Dropout เพื่อให้ผลนิ่ง
    return model

def preprocess_image(img_path):
    # ⚠️ ใช้ Grayscale เหมือนตอน Train!
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"❌ ไม่เจอรูปภาพที่: {img_path}")
    
    # Stack เป็น 3 channels (เหมือน dataset.py)
    img = np.stack([img]*3, axis=-1)
    img = cv2.resize(img, (320, 320))
    
    # เตรียมภาพสำหรับโชว์ (Float 0-1)
    rgb_img_float = np.float32(img) / 255.0

    # เตรียมภาพสำหรับเข้า Model (Normalize แบบเดียวกับตอน Train)
    transform = A.Compose([
        A.Resize(320, 320),
        A.Normalize(mean=[0.485]*3, std=[0.229]*3),  # ⚠️ ตรงกับ dataset.py
        ToTensorV2()
    ])
    
    # Albumentations ต้องการ input เป็น key-value
    tensor = transform(image=img)["image"]
    tensor = tensor.unsqueeze(0).to(DEVICE) # เพิ่ม Batch dimension -> [1, 3, 320, 320]
    
    return rgb_img_float, tensor

def predict_and_visualize(model, rgb_img_float, input_tensor):
    # 1. Prediction
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0] # แปลงเป็น % (0-1)
    
    class_names = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
    
    print("\n📊 --- Prediction Results ---")
    for i, name in enumerate(class_names):
        score = probs[i]
        bar = "█" * int(score * 20)
        print(f"{name:<20}: {score:.4f}  {bar}")

    # 2. Grad-CAM (XAI)
    # เจาะเข้าไปที่ Layer สุดท้ายของ CNN (สำหรับ DenseNet121 คือ features[-1])
    target_layers = [model.model.backbone.features[-1]]
    
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # สร้าง Heatmap
    # targets=None หมายถึงให้โฟกัสที่ Class ที่โมเดลมั่นใจที่สุด
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)
    
    # เอา Heatmap มาแปะทับรูปเดิม
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)

    # 3. Plot รูป
    plt.figure(figsize=(12, 6))
    
    # รูปซ้าย: ต้นฉบับ
    plt.subplot(1, 2, 1)
    plt.imshow(rgb_img_float)
    plt.title("Original X-ray")
    plt.axis("off")
    
    # รูปขวา: XAI Heatmap
    plt.subplot(1, 2, 2)
    plt.imshow(visualization)
    plt.title("AI Attention (Grad-CAM)")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()

# ==========================================
# 🚀 MAIN
# ==========================================
if __name__ == "__main__":
    try:
        # 1. โหลดโมเดล
        model = load_model(CKPT_PATH)
        
        # 2. เตรียมรูป
        rgb_img, tensor = preprocess_image(IMG_PATH)
        
        # 3. ทำนายและโชว์ผล
        predict_and_visualize(model, rgb_img, tensor)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("คำแนะนำ: ตรวจสอบ CKPT_PATH และ IMG_PATH ให้ถูกต้องครับ")