# --- 1. 暴力修复环境缺失 (必须在所有 import 之前) ---
try:
    import pkg_resources
except ImportError:
    import pip._vendor.pkg_resources as pkg_resources
    import sys
    sys.modules['pkg_resources'] = pkg_resources

import os
import face_recognition_models

# --- 2. 强制指定你的模型文件夹位置 ---
# 请确保这 4 个 .dat 文件确实在这个文件夹里
model_path = r'D:\Anaconda\envs\face-attend\Lib\site-packages\face_recognition_models\models'
os.environ['FACE_RECOGNITION_MODELS'] = model_path

# --- 3. 正常导入其他库 ---
import face_recognition
import pickle
import cv2

def build_database(face_db_dir="face_db"):
    print(f"正在扫描人脸库: {face_db_dir}...")
    if not os.path.exists(face_db_dir):
        print(f"错误: 找不到目录 {face_db_dir}")
        return

    known_encodings = []
    known_names = []

    for name in os.listdir(face_db_dir):
        person_dir = os.path.join(face_db_dir, name)
        if not os.path.isdir(person_dir):
            continue
        
        print(f"正在处理: {name}")
        for filename in os.listdir(person_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                filepath = os.path.join(person_dir, filename)
                image = face_recognition.load_image_file(filepath)
                encodings = face_recognition.face_encodings(image)
                if len(encodings) > 0:
                    known_encodings.append(encodings[0])
                    known_names.append(name)

    data = {"encodings": known_encodings, "names": known_names}
    with open("encodings.pkl", "wb") as f:
        f.write(pickle.dumps(data))
    print("\n特征库构建成功！已生成 encodings.pkl")

if __name__ == "__main__":
    build_database()