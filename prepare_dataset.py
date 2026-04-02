"""
机器学习项目一 - 数据集准备工具
================================
功能：
  1. 生成合成人脸样本数据集（用于快速测试流水线）
  2. 使用 OpenCV DNN 自动标注真实图片中的人脸
  3. 自动划分 train/val/test 数据集

用法：
  # 方式一：生成合成测试数据集（快速验证流水线）
  python scripts/prepare_dataset.py --generate 200

  # 方式二：自动标注你自己的图片（推荐）
  python scripts/prepare_dataset.py --auto-label path/to/your/photos/

  # 方式三：下载 WIDER FACE 数据集（需要网络）
  python scripts/prepare_dataset.py --download-wider
"""

import cv2
import numpy as np
import os
import sys
import shutil
import random
import argparse
import urllib.request
import zipfile


# ===================================================================
#  第一部分：生成合成人脸图片（用于流水线测试）
# ===================================================================

def draw_synthetic_face(img, cx, cy, face_w, face_h):
    """在图片上绘制一个合成人脸"""
    # 肤色随机变化
    skin_colors = [
        (180, 200, 230), (160, 190, 220), (140, 175, 210),
        (170, 195, 225), (150, 180, 215), (190, 210, 235),
    ]
    skin = random.choice(skin_colors)

    # 脸部椭圆
    cv2.ellipse(img, (cx, cy), (face_w // 2, face_h // 2), 0, 0, 360, skin, -1)
    cv2.ellipse(img, (cx, cy), (face_w // 2, face_h // 2), 0, 0, 360, 
                (skin[0]-30, skin[1]-30, skin[2]-30), 2)

    # 头发（上半部分）
    hair_colors = [(30, 30, 50), (40, 40, 60), (20, 20, 40), (50, 50, 70), (35, 45, 80)]
    hair = random.choice(hair_colors)
    hair_top = cy - face_h // 2
    cv2.ellipse(img, (cx, hair_top + face_h // 6), 
                (face_w // 2 + 5, face_h // 3), 0, 180, 360, hair, -1)

    # 眼睛
    eye_offset_x = face_w // 5
    eye_y = cy - face_h // 8
    eye_size = max(face_w // 12, 3)
    # 眼白
    cv2.ellipse(img, (cx - eye_offset_x, eye_y), (eye_size + 2, eye_size), 
                0, 0, 360, (240, 240, 245), -1)
    cv2.ellipse(img, (cx + eye_offset_x, eye_y), (eye_size + 2, eye_size), 
                0, 0, 360, (240, 240, 245), -1)
    # 瞳孔
    pupil_colors = [(50, 40, 30), (60, 50, 40), (40, 35, 25)]
    pupil = random.choice(pupil_colors)
    cv2.circle(img, (cx - eye_offset_x, eye_y), eye_size - 1, pupil, -1)
    cv2.circle(img, (cx + eye_offset_x, eye_y), eye_size - 1, pupil, -1)
    # 高光
    cv2.circle(img, (cx - eye_offset_x - 1, eye_y - 1), max(eye_size // 3, 1), 
               (255, 255, 255), -1)
    cv2.circle(img, (cx + eye_offset_x - 1, eye_y - 1), max(eye_size // 3, 1), 
               (255, 255, 255), -1)

    # 眉毛
    brow_y = eye_y - eye_size - 4
    brow_len = face_w // 5
    cv2.line(img, (cx - eye_offset_x - brow_len//2, brow_y),
             (cx - eye_offset_x + brow_len//2, brow_y - 2), hair, 2)
    cv2.line(img, (cx + eye_offset_x - brow_len//2, brow_y - 2),
             (cx + eye_offset_x + brow_len//2, brow_y), hair, 2)

    # 鼻子
    nose_top = cy
    nose_bottom = cy + face_h // 7
    cv2.line(img, (cx, nose_top), (cx - 3, nose_bottom), 
             (skin[0]-20, skin[1]-20, skin[2]-20), 2)
    cv2.line(img, (cx - 3, nose_bottom), (cx + 3, nose_bottom), 
             (skin[0]-20, skin[1]-20, skin[2]-20), 2)

    # 嘴巴
    mouth_y = cy + face_h // 4
    mouth_w = face_w // 4
    if random.random() > 0.5:
        # 微笑
        cv2.ellipse(img, (cx, mouth_y - 3), (mouth_w, mouth_w // 2), 
                    0, 10, 170, (80, 80, 150), 2)
    else:
        # 闭嘴
        cv2.line(img, (cx - mouth_w, mouth_y), (cx + mouth_w, mouth_y), 
                 (100, 100, 160), 2)

    return img


def generate_background(h, w):
    """生成随机背景"""
    bg_type = random.choice(["solid", "gradient", "noise", "scene"])

    if bg_type == "solid":
        color = [random.randint(100, 240) for _ in range(3)]
        img = np.full((h, w, 3), color, dtype=np.uint8)

    elif bg_type == "gradient":
        c1 = np.array([random.randint(80, 220) for _ in range(3)])
        c2 = np.array([random.randint(80, 220) for _ in range(3)])
        img = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(h):
            ratio = y / h
            color = (c1 * (1 - ratio) + c2 * ratio).astype(np.uint8)
            img[y, :] = color

    elif bg_type == "noise":
        base = [random.randint(120, 220) for _ in range(3)]
        img = np.full((h, w, 3), base, dtype=np.uint8)
        noise = np.random.randint(-20, 20, (h, w, 3), dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    else:  # scene-like
        img = np.zeros((h, w, 3), dtype=np.uint8)
        # 天空
        sky = [random.randint(180, 240), random.randint(180, 230), random.randint(100, 160)]
        img[:h//2] = sky
        # 地面
        ground = [random.randint(60, 120), random.randint(80, 140), random.randint(60, 120)]
        img[h//2:] = ground

    return img


def generate_synthetic_dataset(output_dir, num_images=200):
    """生成合成人脸数据集"""

    images_dir = os.path.join(output_dir, "images", "all")
    labels_dir = os.path.join(output_dir, "labels", "all")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    print(f"正在生成 {num_images} 张合成人脸图片...")

    for i in range(num_images):
        # 随机图片尺寸
        img_w = random.randint(400, 800)
        img_h = random.randint(400, 800)

        img = generate_background(img_h, img_w)

        # 随机人脸数量 (1~5)
        num_faces = random.choices([1, 2, 3, 4, 5], weights=[30, 30, 20, 12, 8])[0]

        labels = []
        occupied = []  # 记录已占用区域，避免重叠

        for _ in range(num_faces):
            # 随机人脸大小
            face_h = random.randint(img_h // 6, img_h // 2)
            face_w = int(face_h * random.uniform(0.7, 0.9))

            # 随机位置（确保不超出边界）
            margin = 10
            max_attempts = 20
            placed = False

            for _ in range(max_attempts):
                cx = random.randint(face_w // 2 + margin, img_w - face_w // 2 - margin)
                cy = random.randint(face_h // 2 + margin, img_h - face_h // 2 - margin)

                # 检查是否与已有人脸重叠
                overlap = False
                for (ox, oy, ow, oh) in occupied:
                    if (abs(cx - ox) < (face_w + ow) // 2 and 
                        abs(cy - oy) < (face_h + oh) // 2):
                        overlap = True
                        break

                if not overlap:
                    placed = True
                    break

            if not placed:
                continue

            # 绘制人脸
            draw_synthetic_face(img, cx, cy, face_w, face_h)
            occupied.append((cx, cy, face_w, face_h))

            # 计算 YOLO 标注 (归一化)
            bbox_x = cx / img_w
            bbox_y = cy / img_h
            bbox_w = (face_w + 10) / img_w   # 稍微扩展边界
            bbox_h = (face_h + 10) / img_h
            labels.append(f"0 {bbox_x:.6f} {bbox_y:.6f} {bbox_w:.6f} {bbox_h:.6f}")

        # 添加随机噪声和模糊
        if random.random() > 0.5:
            blur_k = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (blur_k, blur_k), 0)
        if random.random() > 0.7:
            noise = np.random.randint(-10, 10, img.shape, dtype=np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # 随机亮度调整
        if random.random() > 0.5:
            factor = random.uniform(0.7, 1.3)
            img = np.clip(img * factor, 0, 255).astype(np.uint8)

        # 保存
        img_name = f"synth_{i:04d}.jpg"
        cv2.imwrite(os.path.join(images_dir, img_name), img)

        label_name = f"synth_{i:04d}.txt"
        with open(os.path.join(labels_dir, label_name), "w") as f:
            f.write("\n".join(labels))

        if (i + 1) % 50 == 0:
            print(f"  已生成 {i + 1}/{num_images} 张")

    print(f"✓ 合成数据集生成完成: {images_dir}")
    return images_dir, labels_dir


# ===================================================================
#  第二部分：使用 OpenCV 自动标注真实图片
# ===================================================================

def auto_label_images(input_dir, output_dir):
    """使用 OpenCV DNN 人脸检测器自动标注图片"""

    images_dir = os.path.join(output_dir, "images", "all")
    labels_dir = os.path.join(output_dir, "labels", "all")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # 使用 OpenCV 自带的 Haar Cascade 人脸检测器
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        print("错误：无法加载人脸检测器")
        sys.exit(1)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = [f for f in os.listdir(input_dir)
                   if os.path.splitext(f)[1].lower() in exts]

    if not image_files:
        print(f"错误：在 {input_dir} 中未找到图片文件")
        sys.exit(1)

    print(f"正在自动标注 {len(image_files)} 张图片...")
    labeled_count = 0
    face_total = 0

    for img_file in sorted(image_files):
        img_path = os.path.join(input_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 检测人脸
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            continue

        # 生成 YOLO 标注
        labels = []
        for (fx, fy, fw, fh) in faces:
            cx = (fx + fw / 2) / w
            cy = (fy + fh / 2) / h
            bw = fw / w
            bh = fh / h
            labels.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        # 复制图片
        shutil.copy2(img_path, os.path.join(images_dir, img_file))

        # 保存标注
        label_name = os.path.splitext(img_file)[0] + ".txt"
        with open(os.path.join(labels_dir, label_name), "w") as f:
            f.write("\n".join(labels))

        labeled_count += 1
        face_total += len(faces)

    print(f"✓ 自动标注完成: {labeled_count} 张图片, {face_total} 个人脸")
    return images_dir, labels_dir


# ===================================================================
#  第三部分：数据集划分
# ===================================================================

def split_dataset(images_dir, labels_dir, output_base, 
                  train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """将数据集划分为 train/val/test"""

    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_base, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_base, "labels", split), exist_ok=True)

    # 获取所有图片
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = [f for f in os.listdir(images_dir)
              if os.path.splitext(f)[1].lower() in exts]

    random.shuffle(images)
    n = len(images)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:],
    }

    for split_name, file_list in splits.items():
        for img_file in file_list:
            # 复制图片
            src_img = os.path.join(images_dir, img_file)
            dst_img = os.path.join(output_base, "images", split_name, img_file)
            shutil.copy2(src_img, dst_img)

            # 复制标注
            label_file = os.path.splitext(img_file)[0] + ".txt"
            src_label = os.path.join(labels_dir, label_file)
            if os.path.exists(src_label):
                dst_label = os.path.join(output_base, "labels", split_name, label_file)
                shutil.copy2(src_label, dst_label)

        print(f"  {split_name}: {len(file_list)} 张")

    # 清理临时 all 目录
    if os.path.basename(images_dir) == "all":
        shutil.rmtree(os.path.dirname(images_dir).replace("labels", "images") 
                       if "labels" in images_dir else images_dir, ignore_errors=True)
        shutil.rmtree(labels_dir, ignore_errors=True)
        # Clean parent "all" dirs
        for sub in ["images/all", "labels/all"]:
            p = os.path.join(output_base, sub)
            if os.path.exists(p):
                shutil.rmtree(p, ignore_errors=True)

    # 生成 face_data.yaml
    yaml_path = os.path.join(output_base, "face_data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"train: ./{output_base}/images/train\n")
        f.write(f"val: ./{output_base}/images/val\n")
        f.write(f"test: ./{output_base}/images/test\n\n")
        f.write("nc: 1\n")
        f.write("names: ['face']\n")

    print(f"\n✓ 数据集划分完成！")
    print(f"  配置文件: {yaml_path}")


# ===================================================================
#  主函数
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="数据集准备工具")
    parser.add_argument("--generate", type=int, metavar="N",
                        help="生成 N 张合成人脸图片（用于测试流水线）")
    parser.add_argument("--auto-label", type=str, metavar="DIR",
                        help="自动标注指定目录中的真实图片")
    parser.add_argument("--output", type=str, default="dataset",
                        help="输出目录 (默认: dataset)")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.2)

    args = parser.parse_args()

    if not args.generate and not args.auto_label:
        print("请指定操作模式：")
        print("  --generate N       生成 N 张合成人脸图片")
        print("  --auto-label DIR   自动标注 DIR 中的真实图片")
        print("\n示例：")
        print("  python scripts/prepare_dataset.py --generate 200")
        print("  python scripts/prepare_dataset.py --auto-label my_photos/")
        sys.exit(1)

    print("=" * 60)
    print("  机器学习项目一 - 数据集准备工具")
    print("=" * 60)

    if args.generate:
        images_dir, labels_dir = generate_synthetic_dataset(
            args.output, args.generate)
    else:
        images_dir, labels_dir = auto_label_images(
            args.auto_label, args.output)

    # 划分数据集
    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    print(f"\n划分数据集 (train:{args.train_ratio} / val:{args.val_ratio} / test:{test_ratio:.1f})...")
    split_dataset(images_dir, labels_dir, args.output,
                  args.train_ratio, args.val_ratio, test_ratio)

    print("\n" + "=" * 60)
    print("数据集准备完成！接下来可以运行：")
    print("  python scripts/train_yolo.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
