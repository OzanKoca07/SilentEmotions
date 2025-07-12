import numpy as np
import cv2
import mediapipe as mp
import os

mp_face_mesh = mp.solutions.face_mesh
LIPS_LANDMARKS = list(set([pt for pair in mp_face_mesh.FACEMESH_LIPS for pt in pair]))  # Tekil noktalar

def load_video_datas(img, img_size=(128, 128)):
    image = cv2.imread(img)
    if image is None:
        print(f"{img} dosyası okunamadı!")
        return None

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)

        if not results.multi_face_landmarks:
            print("Yüz bulunamadı, döndürülerek tekrar deneniyor...")
            rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            rgb_image = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_image)
            if not results.multi_face_landmarks:
                print(f"{img} dosyasında yüz tespit edilemedi!")
                return None
            image = rotated_image  # Rotated image ile devam et

        landmarks = results.multi_face_landmarks[0]
        h, w, _ = image.shape

        lip_points = []
        for idx in LIPS_LANDMARKS:
            landmark = landmarks.landmark[idx]
            x, y = int(landmark.x * w), int(landmark.y * h)
            lip_points.append((x, y))

        lip_points_np = np.array(lip_points)
        x, y, width, height = cv2.boundingRect(lip_points_np)

        lip = image[y:y+height, x:x+width]
        if lip.size == 0:
            print(f"{img} dudak bölgesi boş çıktı.")
            return None

        lip_resized = cv2.resize(lip, img_size)
        return lip_resized



def create_lips(image_name, output_folder_path, lip):
    if lip is not None:  # Eğer lip boş değilse
        output_image_path = os.path.join(output_folder_path, image_name)
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)  # Eğer output klasörü yoksa oluştur
        cv2.imwrite(output_image_path, lip)
        print(f"{output_image_path} kaydedildi.")
    else:
        print(f"{image_name} için lip tespiti yapılamadı.")

"""
# Görseli işle
lip = load_video_datas("Visual Lip Reading Dataset in Turkish/FULL_Visual_Lip_Reading_Dataset/afiyetolsun/013/01.jpg")

# Eğer lip resmi var ise, oluştur
if lip is not None:
    create_lips("01_lips.jpg", "silentEmotions3/outputs", lip)
else:
    print("Görselden dudak tespiti yapılamadı.")"""

def horizontal_flip(image):
    return cv2.flip(image, 1)



def sigmoid_contrast(image, gain=5, cutoff=0.5):
    normalized_image = image/255.0
    sigmoid = 1 / (1+np.exp(-gain*(normalized_image-cutoff)))
    return np.uint8(sigmoid*255)


def augment_image(image):
    augmented_images = []

    # Yatay çevirme
    augmented_images.append(horizontal_flip(image))

    # Görüntüyü tersine çevirme

    # Sigmoid kontrast
    augmented_images.append(sigmoid_contrast(image, gain=5, cutoff=0.5))

    return augmented_images


def process_all_folders(input_root_folder, output_root_folder, img_size=(128, 128)):
    names = ["horizantal_flip", "sigmoid", "original"]
    for subdir, _, files in os.walk(input_root_folder):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                image_path = os.path.join(subdir, file)
                print(f"İşleniyor: {image_path}")

                lip = load_video_datas(image_path, img_size)
                if lip is not None:
                    relative_path = os.path.relpath(subdir, input_root_folder)
                    actor_name = os.path.basename(subdir)

                    augmented_images = augment_image(lip)
                    augmented_images.append(lip)
                    for i in range(3):
                        name = names[i]
                        img = augmented_images[i]
                        new_folder_name = f"{actor_name}_{name}"
                        new_folder_path = os.path.join(output_root_folder,
                                                       os.path.dirname(relative_path),
                                                       new_folder_name)
                        create_lips(file, new_folder_path, img)
                else:
                    print(f"{image_path} için dudak tespiti yapılamadı.")

if __name__ == "__main__":
    input_root_folder = "Visual Lip Reading Dataset in Turkish/FULL_Visual_Lip_Reading_Dataset"
    output_root_folder = "outputs"
    process_all_folders(input_root_folder, output_root_folder)