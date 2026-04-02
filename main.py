import cv2
import csv
from ultralytics import YOLO
from datetime import datetime
import os

# ===== 1. 配置区域 =====
MODEL_PATH = "yolo11n-face.pt"
# 自动寻找你的 face_db 文件夹下的名字
FACE_DB_DIR = "face_db" 
LOG_DIR = "attendance_log"
CONF_THRES = 0.3  # 降低阈值，让你更容易被检测到

def get_name_from_db():
    """从 face_db 文件夹中获取第一个人的名字作为演示"""
    if os.path.exists(FACE_DB_DIR):
        folders = [f for f in os.listdir(FACE_DB_DIR) if os.path.isdir(os.path.join(FACE_DB_DIR, f))]
        if folders:
            return folders[0] # 返回你文件夹里那个人的名字
    return "Unknown_User"

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 找不到模型文件 {MODEL_PATH}")
        return

    print(f"🚀 正在加载模型并扫描数据库...")
    model = YOLO(MODEL_PATH)
    
    # 自动获取你 face_db 里的名字
    my_name = get_name_from_db()
    print(f"📢 识别目标已设定为: {my_name}")

    attendance_names = set()
    log_records = []

    cap = cv2.VideoCapture(0)
    # 尝试设置较低分辨率以增加流畅度，防止按键卡顿
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("📸 摄像头已开启！")
    print("⚠️ 重要：请确保输入法为【英文】，选中窗口后按【Q】退出！")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        results = model(frame, conf=CONF_THRES, verbose=False)
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                # 签到逻辑
                if my_name not in attendance_names:
                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    attendance_names.add(my_name)
                    log_records.append({"姓名": my_name, "签到时间": now})
                    print(f"✅ {my_name} 签到成功！")

                # 画框和名字
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{my_name} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 统计面板
        cv2.putText(frame, f"Total: {len(attendance_names)}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Face System", frame)

        # 强化版按键检测
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            print("👋 接收到退出指令，正在保存数据...")
            break

    # ===== 4. 导出 CSV (修复时间显示问题) =====
    cap.release()
    cv2.destroyAllWindows()

    if log_records:
        if not os.path.exists(LOG_DIR): 
            os.makedirs(LOG_DIR)
            
        filename = f"attendance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join(LOG_DIR, filename)
        
        # 修复点：确保 fieldnames 与 log_records 的 key 完全一致
        header = ["姓名", "签到时间"]
        
        try:
            with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=header)
                writer.writeheader()
                # 显式写入，确保每一行都包含这两个字段
                for record in log_records:
                    writer.writerow(record)
                    
            print("\n" + "="*40)
            print(f"🎉 任务完成！签到表已生成：\n{os.path.abspath(filepath)}")
            print(f"统计：本次共签到 {len(log_records)} 人")
            print("="*40)
        except Exception as e:
            print(f"❌ 写入文件失败: {e}")

if __name__ == "__main__":
    main()