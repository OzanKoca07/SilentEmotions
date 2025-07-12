import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import os
from PIL import Image
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import random
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns



root_dir = "faces"

# Model hiperparametreleri
NUM_EPOCHS = 50  # Epoch sayısını artırdım
BATCH_SIZE = 4  # Batch boyutunu küçülttüm, daha iyi öğrenme için
LEARNING_RATE = 0.0005  # Öğrenme hızını düşürdüm
EMBED_DIM = 512  # Gömme boyutunu artırdım
NUM_HEADS = 8
NUM_LAYERS = 6  # Transformer katman sayısını artırdım
NUM_OUT_CHANNELS = 256
TRAIN_RATIO = 0.8
RANDOM_SEED = 42
WEIGHT_DECAY = 1e-5  # L2 regularization ekledim
DROPOUT_RATE = 0.3  # Dropout ekledim


def get_transforms():
    # Görüntü büyütme teknikleri eklendi
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    return transform

class EmotionDataset(Dataset):
    def __init__(self, root_folder, transform=None):
        self.root_folder = root_folder
        self.transform = transform
        self.video_paths = []
        self.classes = os.listdir(root_folder)
        self.Frames = []  # (frame_path, emotion_label, coded_folder)

        # "faces" klasörü içindeki tüm duygu klasörlerini bul

        # "faces" içindeki tüm duygu klasörlerini bul (ör. angry, happy)
        for label_idx, emotion in enumerate(self.classes):
            path = os.path.join(root_folder, emotion)
            if os.path.isdir(path):
                for file in os.listdir(path):
                    video_path = os.path.join(path, file)
                    if os.path.isdir(video_path):
                        self.video_paths.append((video_path, label_idx))
                        if transform:
                            self.video_paths.append((video_path, label_idx))

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path, label = self.video_paths[idx]
        frames = []
        for image in sorted(os.listdir(video_path), key=lambda x: str(x.split(".")[0])):
            if image.lower().endswith((".jpg", ".png")):
                frame_path = os.path.join(video_path, image)
                frame = Image.open(frame_path)
                frame = frame.resize((128, 128))
                if frame.mode != "L":
                    frame = frame.convert("L")

                if self.transform and idx % 2 == 0:
                    frame = self.transform(frame)
                else:
                    frame = np.array(frame) / 255.0
                    frame = torch.tensor(frame, dtype=torch.float32).unsqueeze(0)  # Kanal boyutu ekle

                frames.append(frame)

        # Kanal-önce (channel-first) formatında tensörler bekleniyor
        video = torch.stack(frames)  # [T, C, H, W]
        video = video.permute(1, 0, 2, 3)  # [C, T, H, W]

        return video, label

class CNN3D(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(CNN3D, self).__init__()
        self.conv1 = nn.LazyConv3d(out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(64)  # Batch normalization ekledim
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dropout1 = nn.Dropout3d(dropout_rate)

        self.conv2 = nn.LazyConv3d(out_channels=128, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(128)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.dropout2 = nn.Dropout3d(dropout_rate)

        self.conv3 = nn.LazyConv3d(out_channels=256, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.bn3 = nn.BatchNorm3d(256)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.dropout3 = nn.Dropout3d(dropout_rate)

        # Ek konvolüsyon katmanı ekledim
        self.conv4 = nn.LazyConv3d(out_channels=512, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.bn4 = nn.BatchNorm3d(512)
        self.pool4 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dropout4 = nn.Dropout3d(dropout_rate)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)

        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout2(x)

        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout3(x)

        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.dropout4(x)

        return x

def collate_fn(batch):
    videos, labels = zip(*batch)

    max_frames = max([v.shape[1] for v in videos])
    padded_videos = []

    for video in videos:
        if video.shape[1] < max_frames:
            pad_len = max_frames - video.shape[1]
            padding = torch.zeros((video.shape[0], pad_len, video.shape[2], video.shape[3])).to(video.device)
            video = torch.cat((video, padding), dim=1)
        padded_videos.append(video)
    return torch.stack(padded_videos), torch.tensor(labels)



# Eğitim sürecini takip etmek için geçmiş değerlerini kaydedeceğiz
class TrainingHistory:
    def __init__(self):
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []

    def plot(self, save_path=None):
        plt.figure(figsize=(12, 5))

        # Loss grafiği
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()

        # Accuracy grafiği
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accuracies, label='Train Accuracy')
        plt.plot(self.val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)

        plt.show()

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,  # Daha geniş feedforward ağı
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        return self.transformer_encoder(x)

class EmotionModel(nn.Module):
    def __init__(self, cnn_out_channels=512, embed_dim=512, num_heads=8, num_layers=6, num_classes=10,
                 dropout_rate=0.3):
        super(EmotionModel, self).__init__()
        self.cnn3d = CNN3D(dropout_rate=dropout_rate)
        self.temporal_pool = nn.AdaptiveAvgPool3d((20,None, None))

        # Dikkat: CNN çıktı kanallarını değiştirdik, o yüzden linear girdi boyutu değişmeli
        self.fc1 = nn.Linear(cnn_out_channels*8*8, embed_dim)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.transformer = TransformerEncoder(embed_dim, num_heads, num_layers, dropout=dropout_rate)

        self.fc2 = nn.Linear(embed_dim, embed_dim // 2)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(embed_dim // 2, num_classes)

    def feature_extraction(self, x):
        x = self.cnn3d(x)
        x = self.temporal_pool(x)
        B, C, T, H, W = x.shape
        x = x.view(B, C, T, H * W)  # -> (B, C, T, H*W)
        x = x.permute(0, 2, 1, 3)  # -> (B, T, C, H*W)
        x = x.reshape(B, T, C * H * W)  # -> (B, T, input_dim)
        # Fully connected
        x = F.relu(self.fc1(x))
        return x.mean(dim=1)

    def forward(self, x):
        # 3D CNN
        x = self.cnn3d(x)
        # Temporal pooling
        x = self.temporal_pool(x)

        B, C, T, H, W = x.shape
        x = x.view(B, C, T, H * W)  # -> (B, C, T, H*W)
        x = x.permute(0, 2, 1, 3)  # -> (B, T, C, H*W)
        x = x.reshape(B, T, C * H * W)  # -> (B, T, input_dim)
        # Fully connected
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)

        # Transformer
        #x = self.transformer(x)
        #print(x.shape)
        x = x.mean(dim=1)
        # Sınıflandırma
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


def save_model(model, save_path="emotions_reading_model.pth"):
    dir_name = os.path.dirname(save_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model başarıyla kaydedildi: {save_path}")

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs):
    best_val_accuracy = 0.0
    best_model_state = None
    history = TrainingHistory()

    for epoch in range(num_epochs):
        # Eğitim modu
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for video, label in train_loader:
            video, label = video.to(device), label.long().to(device)
            print(f"Video girişi: {video.shape}, Label: {label.shape}")

            optimizer.zero_grad()
            outputs = model(video)
            print(f"Model çıktısı: {outputs.shape}, Label: {label.shape}")
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += label.size(0)
            train_correct += (predicted == label).sum().item()

        train_accuracy = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)

        # Doğrulama modu
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for video, label in val_loader:
                video, label = video.to(device), label.long().to(device)
                outputs = model(video)
                loss = criterion(outputs, label)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += label.size(0)
                val_correct += (predicted == label).sum().item()

        val_accuracy = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        # Scheduler'ı güncelle
        scheduler.step(avg_val_loss)

        # Geçmiş değerlerini kaydet
        history.train_losses.append(avg_train_loss)
        history.train_accuracies.append(train_accuracy)
        history.val_losses.append(avg_val_loss)
        history.val_accuracies.append(val_accuracy)

        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

        # En iyi modeli kaydet
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict()
            torch.save(best_model_state, 'best_model.pt')
            print(">>> Yeni en iyi model kaydedildi.")

    # En iyi modeli geri yükle
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, history

def main():
    dataset = EmotionDataset(root_dir)
    device = "cpu"

    dataset_size = len(dataset)
    train_size = int(TRAIN_RATIO * dataset_size)
    val_size = int(0.1 * dataset_size)  # %10 doğrulama seti
    test_size = dataset_size - train_size - val_size

    train_dataset, val_test_dataset = random_split(
        dataset,
        [train_size, val_size + test_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )

    val_dataset, test_dataset = random_split(
        val_test_dataset,
        [val_size, test_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )

    print(f"Toplam veri sayısı: {dataset_size}")
    print(f"Eğitim veri sayısı: {len(train_dataset)} ({train_size / dataset_size * 100:.1f}%)")
    print(f"Doğrulama veri sayısı: {len(val_dataset)} ({val_size / dataset_size * 100:.1f}%)")
    print(f"Test veri sayısı: {len(test_dataset)} ({test_size / dataset_size * 100:.1f}%)")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True if device == "cuda" else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True if device == "cuda" else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True if device == "cuda" else False
    )

    # Sınıf sayısını belirle
    NUM_CLASSES = len(dataset.classes)
    print(f"Sınıf sayısı: {NUM_CLASSES}")

    # Model oluştur
    model = EmotionModel(
        cnn_out_channels=512,  # CNN çıktı kanallarını değiştirdik
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES,
        dropout_rate=DROPOUT_RATE
    )
    model.to(device)

    # Örnek batch ile model testi
    sample_video, _ = next(iter(train_loader))
    _ = model(sample_video.to(device))

    # Kayıp fonksiyonu, optimizer ve scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )

    # Modeli eğit
    trained_model, history = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        NUM_EPOCHS
    )

    # Sonuçları görselleştir
    os.makedirs("results", exist_ok=True)
    history.plot(save_path="results/training_history_for_emotions.png")

    # Test veri kümesinde değerlendir
    trained_model.eval()
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for video, label in test_loader:
            video, label = video.to(device), label.long().to(device)
            outputs = trained_model(video)

            _, predicted = torch.max(outputs.data, 1)
            test_total += label.size(0)
            test_correct += (predicted == label).sum().item()

    test_accuracy = 100 * test_correct / test_total
    print(f"Test Doğruluk Oranı: {test_accuracy:.2f}%")

    # Confusion matrix hesapla ve görselleştir
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for video, label in test_loader:
            video, label = video.to(device), label.long().to(device)
            outputs = trained_model(video)
            _, predicted = torch.max(outputs.data, 1)

            all_labels.extend(label.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Tahmin')
    plt.ylabel('Gerçek')
    plt.title('Confusion Matrix')
    plt.savefig('results/confusion_matrix_for_emotions.png')

    # Sınıflandırma raporu
    class_names = dataset.classes
    report = classification_report(all_labels, all_predictions, target_names=class_names)
    print("\nSınıflandırma Raporu:")
    print(report)

    with open('results/classification_report_for_emotions.txt', 'w') as f:
        f.write(report)

    if (test_accuracy >75):
        save_model(trained_model,"models/emotions_reading_model.pth")

if __name__ == "__main__":
    main()