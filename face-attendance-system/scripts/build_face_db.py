import face_recognition
import os
import pickle

# 获取当前脚本所在目录的上级目录（项目根目录）
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)

face_db = {}

for person_name in os.listdir('face_db'):
    person_dir = os.path.join('face_db', person_name)
    if not os.path.isdir(person_dir):
        continue

    encodings = []
    for img_file in os.listdir(person_dir):
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(person_dir, img_file)
        print(f'处理: {person_name}/{img_file}')

        image = face_recognition.load_image_file(img_path)
        encs = face_recognition.face_encodings(image)

        if encs:
            encodings.append(encs[0])
            print(f'  ✓ 检测到人脸')
        else:
            print(f'  ✗ 未检测到人脸，建议换一张照片')

    if encodings:
        face_db[person_name] = encodings
        print(f'✓ {person_name}: {len(encodings)} 张照片已入库\n')
    else:
        print(f'✗ {person_name}: 没有有效照片，已跳过\n')

# 保存编码
save_path = os.path.join(project_root, 'encodings.pkl')
with open(save_path, 'wb') as f:
    pickle.dump(face_db, f)
print(f'人脸库构建完成！共 {len(face_db)} 人')
print(f'保存位置：{save_path}')