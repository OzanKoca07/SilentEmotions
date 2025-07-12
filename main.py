import shutil
from datasetConf import load_video_datas
from fastapi import FastAPI, UploadFile, File
import torch
import cv2
from torch.utils.data import Dataset
import os
import numpy as np
from model.multi_modal import FusionModel
from model.dudak_okuma_modeli import LipReadingModel
from model.DuyguModeli import EmotionModel
from model.multi_modal import lip_transform, emotion_transform
from PIL import Image
app = FastAPI()



# Etiket -> Kelime eşleşmesi (outputs klasörüne göre)
id2label = {
0:"hosgeldiniz",
1:
"tesekkur ederim",
2:
"basla",
3:
"bitir",
4:
"ozur dilerim",
5:
"selam",
6:
"gorusmek uzere",
7:
"merhaba",
8:
"gunaydin",
9:
"afiyet olsun",
}

# Model yolları
LIP_MODEL_PATH = "models/improved_lip_reading_model.pth"
EMOTION_MODEL_PATH = "models/emotions_reading_model.pth"
FUSION_MODEL_PATH = "models/multi_model.pth"

FRAME_SIZE = (128, 128)
device = torch.device("cpu")

# Modeller
lip_model = LipReadingModel().to(device)
lip_model.load_state_dict(torch.load(LIP_MODEL_PATH, map_location=device))
lip_model.eval()

# 8 sınıflık duygularla eğitildiği için num_classes = 8
emotion_model = EmotionModel(num_classes=8).to(device)
emotion_model.load_state_dict(torch.load(EMOTION_MODEL_PATH, map_location=device))
emotion_model.eval()

# Fusion model son tahmini 10 kelimeye göre yapacak
fusion_model = FusionModel(
    lip_model,
    emotion_model,
    num_classes=10,  # çünkü kelime sayısı 10
    embed_dim=512
).to(device)
fusion_model.load_state_dict(torch.load(FUSION_MODEL_PATH, map_location=device))
fusion_model.eval()

lip_preprocess = lip_transform()
emotion_preprocess = emotion_transform()

class InferenceEmotionDataset(Dataset):
    def __init__(self, path):
        self.video_paths = path
        self.frames = sorted([f for f in os.listdir(path) if f.lower().endswith((".jpg", ".png"))],
                             key=lambda x: int(x.split('.')[0]))
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        video_path = self.video_paths
        frames1 = []
        frames2 = []

        for image in self.frames:
            if image.lower().endswith((".jpg", ".png")):
                frame_path = os.path.join(video_path, image)
                frame = Image.open(frame_path)
                frame1 = frame.resize((128, 128))
                frame2 = frame.resize((128, 128))

                if frame1.mode != "L":
                    frame1 = frame1.convert("L")
                    frame2 = frame2.convert("L")

                frame1 = np.array(frame1) / 255.0
                frame1 = torch.tensor(frame1, dtype=torch.float32).unsqueeze(0)  # Kanal boyutu ekle

                frames1.append(frame1)

                frame2 = np.array(frame2) / 255.0
                frame2 = torch.tensor(frame2, dtype=torch.float32).unsqueeze(0)  # Kanal boyutu ekle

                frames2.append(frame2)

        # Kanal-önce (channel-first) formatında tensörler bekleniyor
        video1 = torch.stack(frames1)  # [T, C, H, W]
        video1 = video1.permute(1, 0, 2, 3)  # [C, T, H, W]

        video2 = torch.stack(frames2)  # [T, C, H, W]
        video2 = video2.permute(1, 0, 2, 3)  # [C, T, H, W]

        return video1, video2

def ext_fram(video_path, fps=8, output_folder=""):
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    print(frame_rate)
    frame_interval = int(frame_rate // fps) if frame_rate > 0 else 1

    frame_count = 0
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            # BURADA DÖNDÜRÜYORUZ: 90 derece saat yönünde (YATAY -> DİKEY)
            rotated = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)


            frame_filename = os.path.join(output_folder, f"{saved_count}.jpg")
            cv2.imwrite(frame_filename, rotated)
            saved_count += 1

        frame_count += 1

    cap.release()


from fastapi.responses import JSONResponse

@app.post("/predict", response_class=JSONResponse)
async def predict(file: UploadFile = File(...)):
    print(">> /predict endpoint çağrıldı.")
    if file is None:
        print(">> Dosya gelmedi.")
        return {"hata": "Dosya alınamadı."}
    print(">> Gelen dosya:", file.filename)

    with open("temp_video.mp4", "wb") as f:
        f.write(await file.read())

    root = os.getcwd()  # Şu anki çalışma dizini
    file = "temp"  # Dosya adı (örneğin 'temp')

    video_path = os.path.join(root, file)
    ext_fram("temp_video.mp4", output_folder=video_path)

    lip_path = "temp_lip"
    lip_folder_path = os.path.join(root, lip_path)

    if os.path.exists(lip_folder_path):
        shutil.rmtree(lip_folder_path)
    os.makedirs(lip_folder_path)

    for img in os.listdir(video_path):
        img_path = os.path.join(video_path, img)
        lip = load_video_datas(img_path, img_size=(128, 128))

        if lip is not None:
            output_img_path = os.path.join(lip_folder_path, img)  # Dosya adıyla birlikte tam yol
            cv2.imwrite(output_img_path, lip)

    dataset = InferenceEmotionDataset(lip_folder_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

    fusion_model.eval()


    with torch.no_grad():
        for video1, video2 in dataloader:
            output = fusion_model(video1, video2)
            prediction_idx = torch.argmax(output, dim=1).item()
            print(prediction_idx)
            prediction_word = id2label.get(prediction_idx, "bilinmeyen")

    return {"tahmin": prediction_word}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)