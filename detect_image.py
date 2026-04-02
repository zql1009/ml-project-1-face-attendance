"""
机器学习项目一 - 图片签到检测
==============================
功能：对单张/多张图片进行人脸检测+识别+签到
用法：python scripts/detect_image.py --image test.jpg
      python scripts/detect_image.py --dir test_images/
"""

from ultralytics import YOLO
import face_recognition
import cv2
import pickle
import os
import sys
import argparse
import numpy as np

YOLO_MODEL_PATH = "runs/face/exp1/weights/best.pt"
FACE_DB_PATH = "face_db/encodings.pkl"
CONFIDENCE = 0.5
TOLERANCE = 0.5


def detect_and_recognize(image_path, yolo_model, known_names, known_encodings):
    """对单张图片进行检测和识别"""

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"  ✗ 无法读取图片: {image_path}")
        return None, {}

    # YOLO 检测
    results = yolo_model(frame, conf=CONFIDENCE, verbose=False)
    boxes = results[0].boxes

    attendance = {}

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        # 裁剪 + 识别
        face_crop = frame[max(0,y1):min(frame.shape[0],y2),
                          max(0,x1):min(frame.shape[1],x2)]
        name = "Unknown"

        if face_crop.size > 0:
            rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            encs = face_recognition.face_encodings(rgb_crop)
            if encs:
                dists = face_recognition.face_distance(known_encodings, encs[0])
                if len(dists) > 0:
                    best_idx = np.argmin(dists)
                    if dists[best_idx] < TOLERANCE:
                        name = known_names[best_idx]

        # 去重记录
        if name != "Unknown" and name not in attendance:
            attendance[name] = conf

        # 绘制
        color = (0, 200, 0) if name != "Unknown" else (0, 0, 220)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{name} {conf:.2f}"
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # 签到统计
    total = len(attendance)
    cv2.putText(frame, f"Detected: {len(boxes)} faces | Checked in: {total}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    return frame, attendance


def main():
    parser = argparse.ArgumentParser(description="图片人脸签到检测")
    parser.add_argument("--image", type=str, help="单张图片路径")
    parser.add_argument("--dir", type=str, help="图片目录路径")
    parser.add_argument("--output", type=str, default="attendance_log",
                        help="输出目录")
    args = parser.parse_args()

    if not args.image and not args.dir:
        print("请指定 --image 或 --dir 参数")
        sys.exit(1)

    # 加载模型
    yolo_model = YOLO(YOLO_MODEL_PATH)
    with open(FACE_DB_PATH, "rb") as f:
        face_db = pickle.load(f)

    known_names, known_encodings = [], []
    for name, encs in face_db.items():
        for enc in encs:
            known_names.append(name)
            known_encodings.append(enc)

    os.makedirs(args.output, exist_ok=True)

    # 收集图片
    images = []
    if args.image:
        images.append(args.image)
    else:
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        for f in sorted(os.listdir(args.dir)):
            if os.path.splitext(f)[1].lower() in exts:
                images.append(os.path.join(args.dir, f))

    all_attendance = {}

    for img_path in images:
        print(f"\n处理: {img_path}")
        result_frame, attendance = detect_and_recognize(
            img_path, yolo_model, known_names, known_encodings)

        if result_frame is not None:
            out_name = "result_" + os.path.basename(img_path)
            out_path = os.path.join(args.output, out_name)
            cv2.imwrite(out_path, result_frame)
            print(f"  结果保存: {out_path}")

            for name in attendance:
                if name not in all_attendance:
                    all_attendance[name] = img_path

    print(f"\n===== 签到汇总（不重复）=====")
    print(f"签到总人数: {len(all_attendance)}")
    for i, (name, src) in enumerate(all_attendance.items(), 1):
        print(f"  {i}. {name}")


if __name__ == "__main__":
    main()
