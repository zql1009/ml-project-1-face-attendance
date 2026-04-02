"""
WIDER FACE 数据集转 YOLO 格式工具
==================================
功能：将 WIDER FACE 数据集的标注转换为 YOLOv11 可用的 TXT 格式
      并自动划分 train/val 数据集

使用步骤：
  1. 从 http://shuoyang1213.me/WIDERFACE/ 下载以下文件：
     - WIDER_train.zip（训练集图片）
     - WIDER_val.zip（验证集图片）
     - wider_face_split.zip（标注文件）

  2. 解压后目录结构应为：
     wider_face/
     ├── WIDER_train/images/0--Parade/...
     ├── WIDER_val/images/0--Parade/...
     └── wider_face_split/
         ├── wider_face_train_bbx_gt.txt
         └── wider_face_val_bbx_gt.txt

  3. 运行：
     python scripts/convert_wider_face.py --input wider_face/ --output dataset/

用法：
  python scripts/convert_wider_face.py --input wider_face/ --output dataset/
  python scripts/convert_wider_face.py --input wider_face/ --output dataset/ --max-images 500
"""

import os
import sys
import shutil
import argparse
from PIL import Image


def parse_wider_annotation(anno_file):
    """解析 WIDER FACE 标注文件"""
    annotations = {}

    with open(anno_file, "r") as f:
        while True:
            # 读取图片路径
            line = f.readline().strip()
            if not line:
                break

            img_path = line

            # 读取人脸数量
            num_faces = int(f.readline().strip())

            boxes = []
            if num_faces == 0:
                # 有些图片标注为0个人脸但仍有一行数据
                f.readline()
            else:
                for _ in range(num_faces):
                    box_line = f.readline().strip()
                    parts = box_line.split()
                    # WIDER FACE 格式: x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose
                    x1 = int(parts[0])
                    y1 = int(parts[1])
                    w = int(parts[2])
                    h = int(parts[3])
                    invalid = int(parts[7]) if len(parts) > 7 else 0

                    # 跳过无效标注和太小的人脸
                    if invalid == 1 or w < 10 or h < 10:
                        continue

                    boxes.append((x1, y1, w, h))

            if boxes:
                annotations[img_path] = boxes

    return annotations


def convert_to_yolo(x1, y1, w, h, img_w, img_h):
    """将 WIDER FACE 的 (x1, y1, w, h) 转换为 YOLO 格式 (cx, cy, bw, bh)"""
    # 确保坐标不超出图片边界
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w, x1 + w)
    y2 = min(img_h, y1 + h)

    # 计算中心点和宽高（归一化）
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h

    # 确保值在 0~1 范围内
    cx = max(0, min(1, cx))
    cy = max(0, min(1, cy))
    bw = max(0, min(1, bw))
    bh = max(0, min(1, bh))

    return cx, cy, bw, bh


def process_split(split_name, annotations, images_base_dir, output_dir, max_images=None):
    """处理一个数据集分片（train 或 val）"""

    images_out = os.path.join(output_dir, "images", split_name)
    labels_out = os.path.join(output_dir, "labels", split_name)
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)

    count = 0
    total_faces = 0
    skipped = 0

    items = list(annotations.items())
    if max_images:
        items = items[:max_images]

    for img_rel_path, boxes in items:
        # 查找图片文件
        img_full_path = os.path.join(images_base_dir, img_rel_path)

        if not os.path.exists(img_full_path):
            skipped += 1
            continue

        try:
            # 获取图片尺寸
            img = Image.open(img_full_path)
            img_w, img_h = img.size
        except Exception:
            skipped += 1
            continue

        # 转换标注
        yolo_labels = []
        for (x1, y1, w, h) in boxes:
            cx, cy, bw, bh = convert_to_yolo(x1, y1, w, h, img_w, img_h)
            if bw > 0.01 and bh > 0.01:  # 过滤太小的框
                yolo_labels.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        if not yolo_labels:
            skipped += 1
            continue

        # 生成新文件名（扁平化路径）
        flat_name = img_rel_path.replace("/", "_").replace("\\", "_")
        base_name = os.path.splitext(flat_name)[0]

        # 复制图片
        dst_img = os.path.join(images_out, flat_name)
        shutil.copy2(img_full_path, dst_img)

        # 写入标注文件
        dst_label = os.path.join(labels_out, base_name + ".txt")
        with open(dst_label, "w") as f:
            f.write("\n".join(yolo_labels))

        count += 1
        total_faces += len(yolo_labels)

        if count % 100 == 0:
            print(f"  已处理 {count} 张...")

    return count, total_faces, skipped


def main():
    parser = argparse.ArgumentParser(description="WIDER FACE 转 YOLO 格式")
    parser.add_argument("--input", type=str, required=True,
                        help="WIDER FACE 解压后的根目录")
    parser.add_argument("--output", type=str, default="dataset",
                        help="输出目录（默认: dataset）")
    parser.add_argument("--max-images", type=int, default=None,
                        help="每个分片最多处理多少张图片（用于快速测试）")

    args = parser.parse_args()

    print("=" * 60)
    print("  WIDER FACE → YOLO 格式转换工具")
    print("=" * 60)

    # 查找标注文件
    train_anno = os.path.join(args.input, "wider_face_split", "wider_face_train_bbx_gt.txt")
    val_anno = os.path.join(args.input, "wider_face_split", "wider_face_val_bbx_gt.txt")

    if not os.path.exists(train_anno):
        print(f"错误：找不到训练集标注文件 {train_anno}")
        print("请确保目录结构正确：")
        print("  wider_face/")
        print("  ├── WIDER_train/images/...")
        print("  ├── WIDER_val/images/...")
        print("  └── wider_face_split/")
        print("      ├── wider_face_train_bbx_gt.txt")
        print("      └── wider_face_val_bbx_gt.txt")
        sys.exit(1)

    # 处理训练集
    print("\n解析训练集标注...")
    train_annotations = parse_wider_annotation(train_anno)
    print(f"  找到 {len(train_annotations)} 张有效图片")

    train_images_dir = os.path.join(args.input, "WIDER_train", "images")
    print("转换训练集...")
    t_count, t_faces, t_skip = process_split(
        "train", train_annotations, train_images_dir, args.output, args.max_images)
    print(f"  ✓ 训练集: {t_count} 张图片, {t_faces} 个人脸")

    # 处理验证集
    if os.path.exists(val_anno):
        print("\n解析验证集标注...")
        val_annotations = parse_wider_annotation(val_anno)
        print(f"  找到 {len(val_annotations)} 张有效图片")

        val_images_dir = os.path.join(args.input, "WIDER_val", "images")
        max_val = args.max_images // 4 if args.max_images else None
        print("转换验证集...")
        v_count, v_faces, v_skip = process_split(
            "val", val_annotations, val_images_dir, args.output, max_val)
        print(f"  ✓ 验证集: {v_count} 张图片, {v_faces} 个人脸")
    else:
        v_count, v_faces = 0, 0

    # 生成 face_data.yaml
    yaml_path = os.path.join(args.output, "face_data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"train: ./{args.output}/images/train\n")
        f.write(f"val: ./{args.output}/images/val\n")
        f.write(f"\nnc: 1\n")
        f.write(f"names: ['face']\n")

    print("\n" + "=" * 60)
    print("转换完成！")
    print(f"  训练集: {t_count} 张, {t_faces} 个人脸")
    print(f"  验证集: {v_count} 张, {v_faces} 个人脸")
    print(f"  配置文件: {yaml_path}")
    print(f"\n接下来运行：")
    print(f"  python scripts/train_yolo.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
