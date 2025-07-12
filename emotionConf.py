import os
import cv2
import dlib

# Veri seti yolu
dataset_path = "1188976"
frames_output_path = "frames"
faces_output_path = "faces"

# Duygu ID'leri ve karşılık gelen duygu etiketleri
emotions = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"
}


# Duygu etiketleme fonksiyonu
def get_emotion_from_filename(filename):
    try:
        emotion_id = filename.split("-")[2]  # Örn: "01-02-03-01-01-01-01.mp4" -> "03"
        return emotions.get(emotion_id, "unknown")
    except IndexError:
        return "unknown"


# Frame çıkarma fonksiyonu
def extract_frames(video_path, output_folder, fps=5):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(frame_rate // fps) if frame_rate > 0 else 1

    frame_count = 0
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()


# dlib yüz dedektörü ve landmark belirleyici
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def extract_faces(frame_folder, face_output_root, video_filename, target_size=(224, 224)):
    # Videodan duygu etiketini al
    emotion_label = get_emotion_from_filename(video_filename)

    # Üst dizini duyguya göre organize et
    emotion_folder = os.path.join(face_output_root, emotion_label)

    # Aynı duygu altında video ismiyle klasör oluştur
    face_output_folder = os.path.join(emotion_folder, video_filename)
    os.makedirs(face_output_folder, exist_ok=True)

    face_index = 0  # **Her yüz için farklı dosya ismi olması için index**

    for frame_file in sorted(os.listdir(frame_folder)):
        frame_path = os.path.join(frame_folder, frame_file)
        face_image = cv2.imread(frame_path)

        if face_image is None:
            continue

        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:  # **Bir frame içinde birden fazla yüz olabilir**
            x, y, w, h = (face.left(), face.top(), face.width(), face.height())
            face_crop = face_image[y:y + h, x:x + w]

            if face_crop.size == 0:
                continue

            # Yüzü belirlenen boyuta yeniden boyutlandır
            face_resized = cv2.resize(face_crop, target_size)

            face_filename = os.path.join(face_output_folder, f"face_{face_index:04d}.jpg")
            cv2.imwrite(face_filename, face_resized)

            face_index += 1  # **Her yüz için farklı numara**


# Tüm klasörleri dolaş ve işle
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".mp4"):
            video_path = os.path.join(root, file)
            relative_path = os.path.relpath(root, dataset_path)
            video_name = os.path.splitext(file)[0]

            output_folder = os.path.join(frames_output_path, relative_path, video_name)
            extract_frames(video_path, output_folder)

            extract_faces(output_folder, faces_output_path, video_name)


