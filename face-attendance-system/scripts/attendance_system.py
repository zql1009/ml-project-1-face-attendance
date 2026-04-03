from ultralytics import YOLO
import face_recognition
import cv2
import pickle
import datetime
import os

# 切换到项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)

print("=" * 50)
print("人脸检测与识别签到系统")
print("=" * 50)

# ========== 1. 加载模型（使用官方预训练模型） ==========
print("正在加载YOLO模型...")
yolo_model = YOLO('yolo11n.pt')  # 官方预训练模型，无需自己训练
print("✓ YOLO模型加载完成")

# ========== 2. 加载人脸库 ==========
print("正在加载人脸库...")
with open('encodings.pkl', 'rb') as f:
    face_db = pickle.load(f)

known_names = []
known_encodings = []
for name, encs in face_db.items():
    for enc in encs:
        known_names.append(name)
        known_encodings.append(enc)

print(f"✓ 人脸库加载完成，共 {len(known_names)} 个已知人脸")
print(f"  注册人员: {', '.join(set(known_names))}")

# ========== 3. 签到记录 ==========
attendance_record = {}


def mark_attendance(name):
    if name not in attendance_record and name != 'Unknown':
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        attendance_record[name] = now
        print(f'✓ {name} 签到成功！时间: {now}')
        return True
    return False


# ========== 4. 主循环 ==========
print("\n启动摄像头，按 Q 键退出...")
print("按 R 键重置签到记录")
print("-" * 50)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_count = 0
skip_frames = 2  # 每3帧做一次识别，提高流畅度

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("无法获取摄像头画面")
        break

    frame_count += 1

    # 每隔 skip_frames 帧做一次检测和识别
    if frame_count % skip_frames == 0:
        results = yolo_model(frame, conf=0.5, verbose=False)
        boxes = results[0].boxes

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            # 裁剪人脸区域
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            encs = face_recognition.face_encodings(rgb_crop)

            name = 'Unknown'
            if encs:
                matches = face_recognition.compare_faces(known_encodings, encs[0], tolerance=0.5)
                if True in matches:
                    idx = matches.index(True)
                    name = known_names[idx]
                    mark_attendance(name)

            # 绘制框和标签
            color = (0, 255, 0) if name != 'Unknown' else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f'{name} ({conf:.2f})'
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 显示签到人数
    total = len(attendance_record)
    cv2.putText(frame, f'Total Checked In: {total}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow('Face Attendance System', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        attendance_record.clear()
        print('签到记录已重置')

# ========== 5. 保存签到记录 ==========
cap.release()
cv2.destroyAllWindows()

os.makedirs('attendance_log', exist_ok=True)

if attendance_record:
    import pandas as pd

    df = pd.DataFrame([{'Name': k, 'Time': v} for k, v in attendance_record.items()])
    csv_path = f'attendance_log/record_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n签到记录已保存到: {csv_path}")
    print("\n签到名单：")
    for name, time in attendance_record.items():
        print(f"  {name}: {time}")
else:
    print("\n本次无签到记录")

print(f"\n共 {len(attendance_record)} 人签到")
print("=" * 50)