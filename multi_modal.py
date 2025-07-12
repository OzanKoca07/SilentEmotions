from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from torch.optim.lr_scheduler import ReduceLROnPlateau
from triton.language import dtype
from sklearn.utils.class_weight import compute_class_weight

from .dudak_okuma_modeli import LipReadingModel, LipReadingDataset, get_transforms as lip_transform, NUM_HEADS, \
    LEARNING_RATE, WEIGHT_DECAY, train_model
from .DuyguModeli import EmotionModel, EmotionDataset, get_transforms as emotion_transform
import torch.nn as nn
import cv2
import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split, dataset
from tqdm import tqdm
import seaborn as sns



emotion_dataset_path = os.path.join("../outputs")
lip_dataset_path = os.path.join( "../outputs")

TRAIN_RATIO =0.8
RANDOM_SEED = 42
BATCH_SIZE = 4
num_workers = 4
NUM_EPOCHS = 50
EMBEDDING_DIM = 512
NUM_HEADS =8
NUM_LAYERS = 4
NUM_HIDDEN_UNITS = 256
DROPOUT_RATE = 0.4
LEARNING_RATE = 0.0001
NUM_OUT_CHANNELS = 256
WEIGHT_DECAY = 1e-5

class AttentionFusion(nn.Module):
    def __init__(self, embed_dim):
        super(AttentionFusion, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1, x2):
        q = self.query(x1)
        k = self.key(x2)
        v = self.value(x2)
        attn_weights = self.softmax(torch.bmm(q.unsqueeze(1), k.unsqueeze(2)).squeeze(-1))
        fused = attn_weights.unsqueeze(-1) * v
        return fused + x1

class FusionModel(nn.Module):
    def __init__(self, lip_model, emotion_model, num_classes, embed_dim=256, hidden_dim=256, num_heads=4):
        super(FusionModel, self).__init__()
        self.lip_model = lip_model
        self.emotion_model = emotion_model

        # Freeze pre-trained models
        for name, param in self.lip_model.named_parameters():
            if "fc" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        for name, param in self.emotion_model.named_parameters():
            if "fc" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        self.emotion_model.eval()

        # Multi-head attention for fusion
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

        # Fully connected layers after attention
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, lip_data, emotion_data):
        lip_feat = self.lip_model.extract_features(lip_data)        # shape: (B, D)
        emotion_feat = self.emotion_model.feature_extraction(emotion_data)  # shape: (B, D)

        # Prepare input for attention: sequence of 2 vectors per sample
        combined = torch.stack([lip_feat, emotion_feat], dim=1)  # shape: (B, 2, D)

        # Apply multi-head attention (self-attention among lip and emotion features)
        attn_output, _ = self.attn(combined, combined, combined)  # shape: (B, 2, D)

        # Aggregate attended output (e.g., by mean or just the first token)
        fusion_feat = attn_output.mean(dim=1)  # shape: (B, D)

        # Feedforward layers
        x = self.dropout(self.relu(self.fc1(fusion_feat)))
        x = self.fc2(x)
        return x


class FusionDataset(Dataset):
    def __init__(self, lip_dataset, emotion_dataset):
        assert len(lip_dataset) == len(emotion_dataset)
        self.classes = lip_dataset.classes
        self.lip_dataset = lip_dataset
        self.emotion_dataset = emotion_dataset

    def __len__(self):
        return len(self.lip_dataset)

    def __getitem__(self, idx):
        lip_data, label1 = self.lip_dataset[idx]
        emo_data, label2 = self.emotion_dataset[idx]
        assert label1 == label2
        return lip_data, emo_data, label1

def cf(batch):
    lip_videos = []
    emotion_videos = []
    labels = []

    for lip_video, emotion_video, label in batch:
        lip_videos.append(lip_video)
        emotion_videos.append(emotion_video)
        labels.append(label)

    # Lip videolar için padding
    max_lip_frames = max([v.shape[1] for v in lip_videos])
    min_lip_frames = max(4, max_lip_frames)  # minimum 4 frame olacak şekilde zorla
    padded_lip_videos = []
    for video in lip_videos:
        pad_len = min_lip_frames - video.shape[1]
        if pad_len > 0:
            padding = torch.zeros((video.shape[0], pad_len, video.shape[2], video.shape[3]))
            video = torch.cat((video, padding), dim=1)
        padded_lip_videos.append(video)
    padded_lip_videos = torch.stack(padded_lip_videos)

    # Emotion videolar için padding
    max_emotion_frames = max([v.shape[1] for v in emotion_videos])
    min_emotion_frames = max(4, max_emotion_frames)  # minimum 4 frame olacak şekilde zorla
    padded_emotion_videos = []
    for video in emotion_videos:
        pad_len = min_emotion_frames - video.shape[1]
        if pad_len > 0:
            padding = torch.zeros((video.shape[0], pad_len, video.shape[2], video.shape[3]))
            video = torch.cat((video, padding), dim=1)
        padded_emotion_videos.append(video)
    padded_emotion_videos = torch.stack(padded_emotion_videos)

    labels = torch.tensor(labels, dtype=torch.long)
    return padded_lip_videos, padded_emotion_videos, labels





def train(model, dataloader,val_loader,scheduler, optimizer, criterion, device, num_epochs=10):
    best_val_accuracy = 0.0
    best_model_state = None
    early_stop_patience = 10
    epochs_no_improve = 0
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        model.train()

        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for lip_input, emotion_input, labels in loop:
            lip_input = lip_input.to(device)
            emotion_input = emotion_input.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(lip_input, emotion_input)  # logits
            loss = criterion(outputs, labels)  # loss hesapla
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = torch.argmax(outputs, dim=1) # prediction için argmax
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            loop.set_postfix(loss=loss.item(), acc=100.0 * correct / total)

        epoch_loss = total_loss / total
        epoch_acc = 100.0 * correct / total
        print(f"Epoch {epoch+1} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for lip, emotion, label in val_loader:
                lip,emotion, label = lip.to(device), emotion.to(device), label.to(device)
                outputs = model(lip, emotion)
                loss = criterion(outputs, label)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += label.size(0)
                val_correct += (predicted == label).sum().item()

        val_accuracy = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        # Scheduler'ı güncelle
        scheduler.step(avg_val_loss)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict()
            torch.save(best_model_state, 'best_model.pt')
            print(">>> Yeni en iyi model kaydedildi.")
        # Early stopping kontrolü
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(f">>> Early stopping at epoch {epoch + 1}")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model

def save_model(model, save_path="multi_model.pth"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model başarıyla kaydedildi: {save_path}")


def main():
    l_transform = lip_transform()
    d_transform = emotion_transform()

    lip_dataset = LipReadingDataset(lip_dataset_path, l_transform)
    emotion_dataset = EmotionDataset(emotion_dataset_path, d_transform)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(len(lip_dataset), len(emotion_dataset) )
    fusion_dataset = FusionDataset(lip_dataset, emotion_dataset)

    dataset_size = len(fusion_dataset)
    train_size = int(TRAIN_RATIO * dataset_size)
    val_size = int(0.1 * dataset_size)  # %10 doğrulama seti
    test_size = dataset_size - train_size - val_size


    train_dataset, val_test_dataset = random_split(
        fusion_dataset,
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

    labels = [label for _, _, label in train_dataset]
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=cf,
        pin_memory=True if device == "cuda" else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=cf,
        pin_memory=True if device == "cuda" else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=cf,
        pin_memory=True if device == "cuda" else False
    )

    # Sınıf sayısını belirle
    NUM_CLASSES = len(fusion_dataset.classes)
    print(f"Sınıf sayısı: {NUM_CLASSES}")

    lip_model = LipReadingModel()
    lip_model.load_state_dict(torch.load("../models/improved_lip_reading_model.pth"))
    lip_model.eval()

    emotion_model = EmotionModel(num_classes=8)
    emotion_model.load_state_dict(torch.load("../models/emotions_reading_model.pth"))
    emotion_model.eval()

    # Model oluştur
    model = FusionModel(lip_model, emotion_model, num_classes=NUM_CLASSES, embed_dim=512, hidden_dim=256, num_heads=4)
    model.to(device)

    # Örnek batch ile model testi

    # Kayıp fonksiyonu, optimizer ve scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.3,
        patience=7,
        min_lr=1e-6,
        verbose=True
    )

    # Modeli eğit
    trained_model = train(
        model,
        train_loader,
        val_loader,
        scheduler,
        optimizer,
        criterion,
        device,
        NUM_EPOCHS
    )

    # Test veri kümesinde değerlendir
    trained_model.eval()
    test_correct = 0
    test_total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for lip,emotion,label in test_loader:
            lip,emotion, label = lip.to(device),emotion.to(device), label.to(device)
            outputs = trained_model(lip,emotion)

            _, predicted = torch.max(outputs.data, 1)
            test_total += label.size(0)
            test_correct += (predicted == label).sum().item()
            all_labels.extend(label.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())


    test_accuracy = 100 * test_correct / test_total
    print(f"Test Doğruluk Oranı: {test_accuracy:.2f}%")

    # Confusion matrix hesapla ve görselleştir


    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Tahmin')
    plt.ylabel('Gerçek')
    plt.title('Confusion Matrix')
    plt.savefig('../results/confusion_matrix_for_multimodal.png')

    # Sınıflandırma raporu
    class_names = fusion_dataset.classes
    report = classification_report(all_labels, all_predictions, target_names=class_names)
    print("\nSınıflandırma Raporu:")
    print(report)

    with open('../results/classification_report_for_multimodal.txt', 'w') as f:
        f.write(report)

    if (test_accuracy >75):
        save_model(trained_model,"../models/multi_model.pth")

if __name__ == '__main__':
    main()
