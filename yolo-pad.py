import torch
import cv2
from paddleocr import PaddleOCR
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression, scale_coords
import os
import glob
import time
import re

def extract_info_from_texts(texts):
    t_start = time.time()
    joined_text = " ".join(texts)
    print("\n=== 提取结果 ===")

    # 📮 邮政编码（6位纯数字）
    zip_codes = re.findall(r"\b\d{6}\b", joined_text)
    if zip_codes:
        print(f"📮 邮政编码: {zip_codes[0]}")
    else:
        print("📮 未检测到邮政编码")

    # 📦 邮件条码（10-15位数字或大写字母混合）
    barcodes = re.findall(r"\b[A-Z0-9]{10,15}\b", joined_text)
    if barcodes:
        print(f"📦 邮件条码: {barcodes[0]}")
    else:
        print("📦 未检测到邮件条码")

    # 🏠 地址（包含“路”“街”“弄”“巷”“号”“室”等）
    address_lines = [line for line in texts if any(k in line for k in ['路', '街', '弄', '巷', '号', '室']) and len(line) > 6]
    if address_lines:
        print(f"🏠 收件人地址: {' '.join(address_lines)}")
    else:
        print("🏠 未检测到收件人地址")

    # 👤 收件人提取策略
    receiver_lines = []

    # 1. 查找含“收”“先生”“女士”“小姐”
    for line in texts:
        if '收' in line or '先生' in line or '女士' in line or '小姐' in line:
            receiver_lines.append(line)
            break

    # 2. 查找“亲启”或“敬启”前一行
    if not receiver_lines:
        for i, line in enumerate(texts):
            if ('亲启' in line or '敬启' in line) and i > 0:
                prev_line = texts[i - 1]
                if re.match(r'^[\u4e00-\u9fa5]{2,4}$', prev_line):
                    receiver_lines.append(prev_line)
                    break

    if receiver_lines:
        print(f"👤 收件人: {receiver_lines[0]}")
    else:
        print("👤 未检测到收件人")

    print("====================")
    print(f"📊 信息提取耗时: {time.time() - t_start:.3f} 秒\n")


# ========================
#        主程序部分
# ========================

program_start = time.time()


# 1. 加载YOLOv5模型
t_yolo_start = time.time()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = attempt_load('weights/best.pt', map_location=device)
model.eval()
t_yolo_end = time.time()
print(f"✅ YOLO模型加载完成，用时: {t_yolo_end - t_yolo_start:.3f} 秒")

# 2. 加载图片
image_dir = 'train'
image_paths = glob.glob(os.path.join(image_dir, '*.*'))

# 3. 加载OCR模型
t_ocr_start = time.time()
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)
t_ocr_end = time.time()
print(f"✅ OCR模型加载完成，用时: {t_ocr_end - t_ocr_start:.3f} 秒")

# 4. 处理每张图片
total_image_time = 0

for img_path in image_paths:
    print(f"\n📷 正在处理图片: {img_path}")
    total_start = time.time()

    # 读取图片
    im0 = cv2.imread(img_path)
    assert im0 is not None, f'Image Not Found {img_path}'
    img = cv2.resize(im0, (640, 640))
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()
    img = torch.from_numpy(img).float() / 255.0
    img = img.unsqueeze(0).to(device)

    # 阶段1：YOLO检测
    t1 = time.time()
    with torch.no_grad():
        pred = model(img)[0]
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)
    print(f"🔍 YOLO检测耗时: {time.time() - t1:.3f} 秒")

    # 阶段2：裁剪 + OCR识别
    for det in pred:
        if det is not None and len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = map(int, xyxy)
                crop_img = im0[y1:y2, x1:x2]

                os.makedirs('output', exist_ok=True)
                save_path = f"output/crop_{os.path.basename(img_path).split('.')[0]}_{x1}_{y1}_{x2}_{y2}.jpg"
                cv2.imwrite(save_path, crop_img)

                t2 = time.time()
                result = ocr.predict(crop_img)
                ocr_time = time.time() - t2
                print(f"🧾 OCR识别耗时: {ocr_time:.3f} 秒")

                if result and isinstance(result, list) and 'rec_texts' in result[0]:
                    texts = result[0]['rec_texts']
                    for text in texts:
                        print(text)
                    extract_info_from_texts(texts)
                else:
                    print("未识别到文字")

    per_image_time = time.time() - total_start
    total_image_time += per_image_time
    print(f"📌 图片总处理耗时: {per_image_time:.3f} 秒")

# 5. 程序总耗时
program_end = time.time()
print("\n✅ 所有图片处理完成！")
print(f"🕒 总图片数: {len(image_paths)}")
print(f"🕒 总图片处理耗时: {total_image_time:.3f} 秒")
print(f"🕒 程序总运行时间: {program_end - program_start:.3f} 秒")
