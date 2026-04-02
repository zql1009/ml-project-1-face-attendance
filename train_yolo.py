"""
机器学习项目一 - Step 1: YOLOv11 人脸检测模型训练
=================================================
功能：使用 YOLOv11 训练人脸检测模型
用法：python scripts/train_yolo.py
"""

from ultralytics import YOLO
import os

def main():
    # ===== 配置区域（根据需要修改）=====
    MODEL_SIZE = "yolo11n.pt"       # 可选：yolo11n.pt / yolo11s.pt / yolo11m.pt
    DATA_YAML = "dataset/face_data.yaml"
    EPOCHS = 80
    IMG_SIZE = 480
    BATCH_SIZE = 8                 # 显存不足请改为 8 或 4
    DEVICE = 'cpu'                     # GPU编号，无GPU填 'cpu'
    PROJECT = "runs/face"
    NAME = "exp1"
    PATIENCE = 20                   # 早停耐心值

    # ===== 检查数据集 =====
    if not os.path.exists(DATA_YAML):
        print(f"错误：找不到数据集配置文件 {DATA_YAML}")
        print("请先准备数据集并创建 face_data.yaml")
        return

    # ===== 加载预训练模型 =====
    print(f"加载预训练模型: {MODEL_SIZE}")
    model = YOLO(MODEL_SIZE)

    # ===== 开始训练 =====
    print("开始训练...")
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project=PROJECT,
        name=NAME,
        patience=PATIENCE,
        save=True,
        plots=True,
        verbose=True,
        workers=0,
    )

    print(f"\n训练完成！")
    print(f"最佳模型保存在: {results.save_dir}/weights/best.pt")
    print(f"训练曲线图保存在: {results.save_dir}/")

    # ===== 自动评估 =====
    print("\n开始评估模型...")
    best_model = YOLO(f"{results.save_dir}/weights/best.pt")
    metrics = best_model.val(data=DATA_YAML)

    print(f"\n===== 评估结果 =====")
    print(f"mAP@0.5     : {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"Precision   : {metrics.box.mp:.4f}")
    print(f"Recall      : {metrics.box.mr:.4f}")

    if metrics.box.map50 >= 0.75:
        print("\n✓ mAP@0.5 达标（≥0.75），模型训练成功！")
    else:
        print("\n✗ mAP@0.5 未达标（<0.75），建议：")
        print("  1. 增加数据集数量和多样性")
        print("  2. 增加训练 epochs")
        print("  3. 尝试更大的模型（yolo11s / yolo11m）")


if __name__ == "__main__":
    main()
