import torch
import torch.nn as nn
import cv2
import json
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm
import os
import shutil
from ultralytics import YOLO
import yaml
import time
import pandas as pd

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.ops import nms, box_iou

# ==============================================================================
# 1. KIẾN TRÚC ATTENTION U-NET
# ==============================================================================
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi
class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)
        out = self.gamma * out + x
        return out
class AttentionUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 256)
        self.conv4 = ConvBlock(256, 512)
        self.bottleneck = ConvBlock(512, 1024)
        self.sa = SelfAttention(1024)
        self.up_conv5 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.att5 = AttentionGate(F_g=512, F_l=512, F_int=256)
        self.dec_conv5 = ConvBlock(1024, 512)
        self.up_conv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att4 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.dec_conv4 = ConvBlock(512, 256)
        self.up_conv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att3 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.dec_conv3 = ConvBlock(256, 128)
        self.up_conv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att2 = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.dec_conv2 = ConvBlock(128, 64)
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(self.pool(x1))
        x3 = self.conv3(self.pool(x2))
        x4 = self.conv4(self.pool(x3))
        bottleneck = self.bottleneck(self.pool(x4))
        sa_out = self.sa(bottleneck)
        d5 = self.up_conv5(sa_out)
        x4_att = self.att5(g=d5, x=x4)
        d5 = torch.cat((x4_att, d5), dim=1)
        d5 = self.dec_conv5(d5)
        d4 = self.up_conv4(d5)
        x3_att = self.att4(g=d4, x=x3)
        d4 = torch.cat((x3_att, d4), dim=1)
        d4 = self.dec_conv4(d4)
        d3 = self.up_conv3(d4)
        x2_att = self.att3(g=d3, x=x2)
        d3 = torch.cat((x2_att, d3), dim=1)
        d3 = self.dec_conv3(d3)
        d2 = self.up_conv2(d3)
        x1_att = self.att2(g=d2, x=x1)
        d2 = torch.cat((x1_att, d2), dim=1)
        d2 = self.dec_conv2(d2)
        return self.out_conv(d2)

# ==============================================================================
# 2. Function Feature
# ==============================================================================
def load_unet_model(path, num_classes, device):
    model = AttentionUNet(in_channels=3, out_channels=num_classes).to(device)
    checkpoint = torch.load(path, map_location=torch.device(device))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model

def prepare_evaluation_set_from_coco(coco_root, dest_dir, split_configs):
    """Tạo tập đánh giá từ dataset COCO gốc dựa trên các quy tắc lọc."""
    dest_dir = Path(dest_dir)
    if dest_dir.exists():
        print(f"Thư mục đánh giá '{dest_dir}' đã tồn tại. Sẽ sử dụng lại.")
        # Tái tạo lại danh sách file từ thư mục đã có
        evaluation_sets = {}
        for split in split_configs.keys():
            split_img_dir = dest_dir / split
            if split_img_dir.is_dir():
                image_paths = sorted(list(split_img_dir.glob('*.jpg')))
                json_path = coco_root / split / '_annotations.coco.json'
                with open(json_path, 'r') as f: data = json.load(f)
                img_name_to_id = {img['file_name']: img['id'] for img in data['images']}
                annotations_by_image = {img['id']: [] for img in data['images']}
                for ann in data['annotations']: annotations_by_image[ann['image_id']].append(ann)
                
                split_data = []
                for img_path in image_paths:
                    img_id = img_name_to_id.get(img_path.name)
                    if img_id is not None:
                        gt_count = len(annotations_by_image.get(img_id, []))
                        split_data.append((img_path, gt_count))
                evaluation_sets[split] = split_data
        return evaluation_sets

    print(f"Đang tạo tập đánh giá tại: {dest_dir}")
    evaluation_sets = {}

    for split, config in split_configs.items():
        print(f"\n--- Đang lọc dữ liệu từ split: {split} ---")
        json_path = coco_root / split / '_annotations.coco.json'
        if not json_path.exists():
            print(f"Cảnh báo: Không tìm thấy {json_path}. Bỏ qua split này.")
            continue
            
        with open(json_path, 'r') as f: data = json.load(f)
        
        img_id_to_info = {img['id']: img for img in data['images']}
        annotations_by_image = {img_id: [] for img_id in img_id_to_info.keys()}
        for ann in data['annotations']:
            # Chỉ đếm các annotation có bbox (để đảm bảo là một vật thể)
            if 'bbox' in ann:
                annotations_by_image[ann['image_id']].append(ann)
        
        filtered_images = []
        for img_id, anns in annotations_by_image.items():
            if len(anns) >= config['min_objects']:
                img_info = img_id_to_info[img_id]
                img_info['gt_count'] = len(anns)
                filtered_images.append(img_info)
        
        print(f"Tìm thấy {len(filtered_images)} ảnh thỏa mãn điều kiện (>= {config['min_objects']} vật thể).")

        if config['num_images'] is not None and len(filtered_images) > config['num_images']:
            selected_images = random.sample(filtered_images, config['num_images'])
            print(f"Đã lấy ngẫu nhiên {len(selected_images)} ảnh.")
        else:
            selected_images = filtered_images
            print(f"Sử dụng tất cả {len(selected_images)} ảnh đã lọc.")
            
        split_dest_dir = dest_dir / split
        split_dest_dir.mkdir(parents=True, exist_ok=True)
        split_data = []
        for img_info in tqdm(selected_images, desc=f"Copying {split} images"):
            source_path = coco_root / split / img_info['file_name']
            dest_path = split_dest_dir / img_info['file_name']
            if source_path.exists():
                shutil.copy(source_path, dest_path)
                split_data.append((dest_path, img_info['gt_count']))
        
        evaluation_sets[split] = split_data
        
    return evaluation_sets

def draw_results_on_image(image, gt, pure_count, pipeline_count, pipeline_boxes, unet_mask, class_names):
    """Vẽ kết quả đếm, bounding box và mask lên ảnh."""
    # Phủ lớp mask U-Net (màu xanh lá)
    color_mask = np.zeros_like(image, dtype=np.uint8)
    color_mask[unet_mask > 0] = [0, 255, 0] # Green
    image = cv2.addWeighted(image, 1.0, color_mask, 0.3, 0)
    
    # Vẽ các bounding box của pipeline (màu magenta)
    for box_data in pipeline_boxes:
        x1, y1, x2, y2 = box_data['box']
        class_id = box_data['class_id']
        # SỬA LỖI: Chuyển đổi class_name thành string một cách an toàn
        class_name = str(class_names.get(class_id, f'ID:{class_id}'))
        
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 3)
        cv2.putText(image, class_name, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

    # Vẽ text thống kê
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 2
    texts = [
        f"Ground Truth: {gt}",
        f"YOLO Pure: {pure_count}",
        f"Pipeline (Hybrid): {pipeline_count}"
    ]
    colors = [(255, 255, 255), (0, 150, 255), (255, 0, 255)]

    for i, (text, color) in enumerate(zip(texts, colors)):
        y = (i + 1) * 40 + 10
        cv2.putText(image, text, (15, y), font_face, font_scale, (0,0,0), thickness + 4)
        cv2.putText(image, text, (15, y), font_face, font_scale, color, thickness)
        
    return image

# ==============================================================================
# 3. HÀM MAIN ĐỂ SO SÁNH
# ==============================================================================
def compare_counting_methods():
    # --- CẤU HÌNH ---
    YOLO_MODEL_PATH = '/home/dangnguyen/dev/yolo/runs/detect/yolov11_final_run_with_tuned_params/weights/best.pt'
    UNET_MODEL_PATH = "/home/dangnguyen/dev/yolo/unet_best.pth.tar"
    COCO_DATASET_ROOT = Path('/home/dangnguyen/dev/yolo/dataset')
    
    EVALUATION_SET_DIR = Path("counting_evaluation_set_from_coco")
    OUTPUT_DIR = Path("final_counting_report_hybrid_pipeline")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    split_configs = {
        'train': {'num_images': 200, 'min_objects': 10},
        'valid': {'num_images': 100, 'min_objects': 1},
        'test':  {'num_images': None, 'min_objects': 0}
    }
    
    # --- CẤU HÌNH PIPELINE ---
    YOLO_CONF_THRESHOLD = 0.25
    IMAGE_SIZE_UNET = 256
    MASK_CONFIRM_THRESHOLD = 0.25
    NMS_IOU_THRESHOLD = 0.4
    UNET_MISSED_IOU_THRESHOLD = 0.1
    
    # --- SETUP ---
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "visuals").mkdir(exist_ok=True, parents=True)
    
    evaluation_sets = prepare_evaluation_set_from_coco(COCO_DATASET_ROOT, EVALUATION_SET_DIR, split_configs)
    if not any(evaluation_sets.values()): return

    # --- TẢI CÁC MODEL ---
    print("\nĐang tải các model...")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    with open(COCO_DATASET_ROOT / 'train' / '_annotations.coco.json', 'r') as f: coco_data = json.load(f)
    valid_categories = [cat for cat in coco_data.get('categories', []) if cat.get('supercategory') != 'none']
    unet_num_classes = len(valid_categories) + 1
    # SỬA LỖI: Lấy tên lớp trực tiếp từ model YOLO
    yolo_class_names = yolo_model.names
    unet_model = load_unet_model(UNET_MODEL_PATH, unet_num_classes, DEVICE)
    
    unet_transform = A.Compose([
        A.Resize(height=IMAGE_SIZE_UNET, width=IMAGE_SIZE_UNET),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ])
    
    results_data = []
    
    for split, image_data in evaluation_sets.items():
        if not image_data: continue
        print(f"\nBắt đầu so sánh trên {len(image_data)} ảnh từ tập '{split}'...")
        for img_path, gt_count in tqdm(image_data, desc=f"Comparing on {split}"):
            original_image = cv2.imread(str(img_path))
            if original_image is None: continue
            h, w, _ = original_image.shape

            # --- CHẠY SONG SONG 2 MODEL ---
            yolo_pure_results = yolo_model.predict(original_image, verbose=False, conf=YOLO_CONF_THRESHOLD)
            
            unet_input_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            unet_input_tensor = unet_transform(image=unet_input_img)["image"].unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                unet_pred_mask_resized = torch.argmax(unet_model(unet_input_tensor), dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            unet_mask_full = cv2.resize(unet_pred_mask_resized, (w, h), interpolation=cv2.INTER_NEAREST)

            # --- 1. PHƯƠNG PHÁP YOLO THUẦN TÚY ---
            yolo_pure_count = len(yolo_pure_results[0].boxes)
            
            # --- 2. PIPELINE "HYBRID" ---
            # Bước 2.1: Lọc bỏ False Positives (YOLO -> U-Net -> NMS)
            validated_candidates = []
            for box in yolo_pure_results[0].boxes:
                x1, y1, x2, y2 = box.xyxy.cpu().numpy().flatten().astype(int)
                box_region_on_unet_mask = unet_mask_full[y1:y2, x1:x2]
                if box_region_on_unet_mask.size > 0:
                    confirmation_score = np.mean(box_region_on_unet_mask > 0)
                else:
                    confirmation_score = 0
                
                if confirmation_score > MASK_CONFIRM_THRESHOLD:
                    validated_candidates.append({
                        "box": [x1, y1, x2, y2],
                        "score": box.conf.cpu().item(),
                        "class_id": int(box.cls.cpu().item())
                    })

            if validated_candidates:
                boxes_to_nms = torch.tensor([item["box"] for item in validated_candidates], dtype=torch.float32)
                scores_to_nms = torch.tensor([item["score"] for item in validated_candidates], dtype=torch.float32)
                keep_indices = nms(boxes_to_nms, scores_to_nms, iou_threshold=NMS_IOU_THRESHOLD)
                validated_yolo_boxes = [validated_candidates[i] for i in keep_indices]
            else:
                validated_yolo_boxes = []

            # Bước 2.2: Tìm kiếm False Negatives (U-Net -> YOLO)
            unet_contours, _ = cv2.findContours((unet_mask_full > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            missed_by_yolo_count = 0
            if unet_contours:
                validated_boxes_tensor = torch.tensor([item["box"] for item in validated_yolo_boxes], dtype=torch.float32) if validated_yolo_boxes else torch.empty((0, 4))
                
                for cnt in unet_contours:
                    x, y, cw, ch = cv2.boundingRect(cnt)
                    unet_contour_box = torch.tensor([[x, y, x + cw, y + ch]], dtype=torch.float32)
                    
                    if validated_yolo_boxes:
                        overlap = box_iou(unet_contour_box, validated_boxes_tensor)
                        if torch.max(overlap) < UNET_MISSED_IOU_THRESHOLD:
                            missed_by_yolo_count += 1
                    else:
                        missed_by_yolo_count += 1
            
            # Bước 2.3: Tổng hợp kết quả
            pipeline_final_count = len(validated_yolo_boxes) + missed_by_yolo_count

            # --- LƯU KẾT QUẢ ---
            results_data.append({
                'image': img_path.name, 'split': split, 'gt_count': gt_count,
                'yolo_pure_count': yolo_pure_count, 
                'pipeline_final_count': pipeline_final_count,
            })

            # --- VẼ ẢNH SO SÁNH TRỰC QUAN ---
            viz_image = draw_results_on_image(original_image.copy(), gt_count, yolo_pure_count, pipeline_final_count, validated_yolo_boxes, unet_mask_full, yolo_class_names)
            (OUTPUT_DIR / "visuals" / split).mkdir(exist_ok=True, parents=True)
            cv2.imwrite(str(OUTPUT_DIR / "visuals" / split / f"compare_{img_path.name}"), viz_image)

    # --- BÁO CÁO KẾT QUẢ ---
    if not results_data: return
    df = pd.DataFrame(results_data)
    df.to_csv(OUTPUT_DIR / "counting_comparison_metrics_hybrid.csv", index=False)
    
    # Báo cáo tổng thể
    yolo_overall_acc = (df['yolo_pure_count'] == df['gt_count']).mean() * 100
    pipeline_overall_acc = (df['pipeline_final_count'] == df['gt_count']).mean() * 100
    yolo_overall_mae = (df['yolo_pure_count'] - df['gt_count']).abs().mean()
    pipeline_overall_mae = (df['pipeline_final_count'] - df['gt_count']).abs().mean()
    
    overall_summary_data = {
        "Metric": ["Overall Accuracy (%)", "Overall MAE"],
        "YOLO Pure": [yolo_overall_acc, yolo_overall_mae],
        "Hybrid Pipeline": [pipeline_overall_acc, pipeline_overall_mae]
    }
    overall_summary_df = pd.DataFrame(overall_summary_data).set_index("Metric")
    print("\n" + "="*60)
    print(" " * 15 + "KẾT QUẢ ĐÁNH GIÁ TỔNG THỂ")
    print("="*60)
    print(overall_summary_df.round(4))
    print("="*60)
    
if __name__ == "__main__":
    compare_counting_methods()
