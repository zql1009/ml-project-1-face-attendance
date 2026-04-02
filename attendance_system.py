"""
机器学习项目一 - Step 3: 人脸签到系统（核心程序）
=================================================
功能：
  1. 使用 YOLOv11 检测人脸
  2. 使用 face_recognition 识别人脸并报出姓名
  3. 统计签到总人数，不重复计数
  4. 实时显示边界框、姓名、置信度、签到总人数
  5. 退出后导出签到记录为 CSV

用法：python scripts/attendance_system.py
按 'q' 键退出程序

前置条件：
  1. 已训练好 YOLO 人脸检测模型（best.pt）
  2. 已构建人脸特征库（encodings.pkl）
"""

from ultralytics import YOLO
import face_recognition
import cv2
import pickle
import datetime
import os
import sys
import numpy as np
import pandas as pd


# ===== 配置区域 =====
YOLO_MODEL_PATH = "yolo11n-face.pt"  # YOLO 模型路径
FACE_DB_PATH = "encodings.pkl"               # 人脸特征库路径
CAMERA_ID = 0                                          # 摄像头编号（0=默认）
CONFIDENCE_THRESHOLD = 0.5                             # YOLO 检测置信度阈值
RECOGNITION_TOLERANCE = 0.5                            # 人脸识别容忍度（越小越严格）
OUTPUT_DIR = "attendance_log"                           # 签到记录输出目录
SKIP_FRAMES = 2                                        # 每隔N帧做一次识别（提升FPS）


def load_models():
    """加载 YOLO 模型和人脸特征库"""

    # 加载 YOLO 模型
    if not os.path.exists(YOLO_MODEL_PATH):
        print(f"错误：找不到 YOLO 模型文件 {YOLO_MODEL_PATH}")
        print("请先运行 python scripts/train_yolo.py 训练模型")
        sys.exit(1)

    print(f"加载 YOLO 模型: {YOLO_MODEL_PATH}")
    yolo_model = YOLO(YOLO_MODEL_PATH)

    # 加载人脸特征库
    if not os.path.exists(FACE_DB_PATH):
        print(f"错误：找不到人脸特征库 {FACE_DB_PATH}")
        print("请先运行 python scripts/build_face_db.py 构建特征库")
        sys.exit(1)

    print(f"加载人脸特征库: {FACE_DB_PATH}")
    with open(FACE_DB_PATH, "rb") as f:
        face_db = pickle.load(f)

    # 展开为列表供匹配使用
    known_names = []
    known_encodings = []
    for name, encs in face_db.items():
        for enc in encs:
            known_names.append(name)
            known_encodings.append(enc)

    print(f"已加载 {len(face_db)} 人的人脸特征")
    return yolo_model, known_names, known_encodings


def recognize_face(face_crop, known_names, known_encodings):
    """对裁剪出的人脸区域进行识别，返回姓名"""

    if face_crop.size == 0:
        return "Unknown"

    # 转为 RGB（face_recognition 需要 RGB 格式）
    rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

    # 编码人脸
    encodings = face_recognition.face_encodings(rgb_crop)
    if not encodings:
        return "Unknown"

    # 与已知人脸比对
    face_distances = face_recognition.face_distance(known_encodings, encodings[0])

    if len(face_distances) > 0:
        best_idx = np.argmin(face_distances)
        if face_distances[best_idx] < RECOGNITION_TOLERANCE:
            return known_names[best_idx]

    return "Unknown"


def draw_results(frame, boxes_info, attendance_record):
    """在画面上绘制检测结果和签到信息"""

    for (x1, y1, x2, y2, name, conf) in boxes_info:
        # 已识别 = 绿色，未识别 = 红色
        if name != "Unknown":
            color = (0, 200, 0)
            label = f"{name} {conf:.2f}"
        else:
            color = (0, 0, 220)
            label = f"Unknown {conf:.2f}"

        # 绘制边界框
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # 绘制姓名标签背景
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                      (x1 + label_size[0], y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # ===== 左上角：签到统计面板 =====
    total = len(attendance_record)
    panel_h = 40 + total * 25 + 10
    panel_h = max(panel_h, 50)

    # 半透明背景
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (300, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # 标题
    cv2.putText(frame, f"Checked In: {total}",
                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # 签到名单
    for i, (name, time_str) in enumerate(attendance_record.items()):
        short_time = time_str.split(" ")[1] if " " in time_str else time_str
        text = f"{i+1}. {name} ({short_time})"
        cv2.putText(frame, text, (15, 55 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return frame


def export_attendance(attendance_record):
    """导出签到记录为 CSV"""

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"attendance_{date_str}.csv"
    filepath = os.path.join(OUTPUT_DIR, filename)

    records = [{"序号": i+1, "姓名": name, "签到时间": time}
               for i, (name, time) in enumerate(attendance_record.items())]

    df = pd.DataFrame(records)
    df.to_csv(filepath, index=False, encoding="utf-8-sig")

    print(f"\n签到记录已导出: {filepath}")
    print(df.to_string(index=False))
    return filepath


def main():
    """主函数：运行人脸签到系统"""

    print("=" * 60)
    print("  机器学习项目一 - 人脸检测与识别签到系统")
    print("  按 'q' 退出  |  按 's' 手动截图保存")
    print("=" * 60)

    # 加载模型
    yolo_model, known_names, known_encodings = load_models()

    # 签到记录（去重：同一人只记录第一次）
    attendance_record = {}  # { '姓名': '签到时间' }

    # 打开摄像头
    print(f"\n打开摄像头 {CAMERA_ID}...")
    cap = cv2.VideoCapture(CAMERA_ID)

    if not cap.isOpened():
        print("错误：无法打开摄像头！")
        print("如果使用视频文件测试，请修改 CAMERA_ID 为视频路径")
        sys.exit(1)

    print("摄像头已打开，开始检测...\n")

    frame_count = 0
    last_boxes_info = []  # 缓存上一次的识别结果

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # ===== YOLO 检测人脸 =====
        results = yolo_model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        boxes = results[0].boxes

        # 每隔 N 帧做一次完整的识别（其余帧只用 YOLO 检测）
        if frame_count % (SKIP_FRAMES + 1) == 0 or frame_count == 1:
            boxes_info = []

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                # 裁剪人脸区域
                face_crop = frame[max(0, y1):min(frame.shape[0], y2),
                                  max(0, x1):min(frame.shape[1], x2)]

                # 识别人脸
                name = recognize_face(face_crop, known_names, known_encodings)

                # 签到（去重）
                if name != "Unknown" and name not in attendance_record:
                    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    attendance_record[name] = now
                    print(f"✓ 签到成功: {name}  时间: {now}  "
                          f"当前总人数: {len(attendance_record)}")

                boxes_info.append((x1, y1, x2, y2, name, conf))

            last_boxes_info = boxes_info
        else:
            # 非识别帧：只用 YOLO 的检测框，姓名用缓存
            boxes_info = []
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                # 尝试匹配缓存的姓名（按位置最近匹配）
                name = "..."
                for (lx1, ly1, lx2, ly2, lname, _) in last_boxes_info:
                    if abs(x1-lx1) < 50 and abs(y1-ly1) < 50:
                        name = lname
                        break
                boxes_info.append((x1, y1, x2, y2, name, conf))

        # 绘制结果
        frame = draw_results(frame, boxes_info, attendance_record)

        # 显示画面
        cv2.imshow("Face Attendance System - Press 'q' to quit", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            # 手动截图
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(OUTPUT_DIR, f"screenshot_{ts}.jpg")
            cv2.imwrite(save_path, frame)
            print(f"截图已保存: {save_path}")

    # 清理
    cap.release()
    cv2.destroyAllWindows()

    # ===== 导出签到记录 =====
    print("\n" + "=" * 60)
    if attendance_record:
        print(f"签到完成！共 {len(attendance_record)} 人签到（不重复）")
        export_attendance(attendance_record)
    else:
        print("本次无人签到")
    print("=" * 60)


if __name__ == "__main__":
    main()
