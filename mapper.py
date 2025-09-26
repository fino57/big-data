#!/usr/bin/env python
import sys
import os
import io
import subprocess
from PIL import Image
import torch
import numpy as np

# Thêm thư mục 'conch' đã được giải nén vào Python Path
sys.path.insert(0, "./conch") 
from open_clip_custom.factory import create_model_from_pretrained

Image.MAX_IMAGE_PIXELS = None # Cho phép PIL mở các ảnh rất lớn

# ===================================================================
# LỚP FeatureExtractor
# ===================================================================
class FeatureExtractor:
    def __init__(self, ckpt_path="ckpts/conch.pth", device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        model_cfg = 'conch_ViT-B-16'
        self.model, self.preprocess = create_model_from_pretrained(
            model_cfg, ckpt_path, device=self.device
        )
        self.model.eval()

    def extract_features_from_tile(self, tile_image):
        """Trích xuất đặc trưng từ một vùng ảnh (tile) đã được crop."""
        try:
            tensor = self.preprocess(tile_image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = self.model.encode_image(tensor)
                return feat.squeeze(0).cpu().numpy()
        except Exception as e:
            print(f"⚠️ Lỗi khi xử lý tile: {str(e)}", file=sys.stderr)
            return None

# ===================================================================
# LOGIC CHÍNH CỦA MAPPER
# ===================================================================

print(">>> Mapper: Bắt đầu khởi tạo FeatureExtractor...", file=sys.stderr)
try:
    # Model checkpoint được gửi qua -files và sẽ có ở thư mục làm việc hiện tại
    extractor = FeatureExtractor(ckpt_path="./conch.pth")
    print(">>> Mapper: Khởi tạo FeatureExtractor thành công.", file=sys.stderr)
except Exception as e:
    print(f"!!! LỖI MAPPER: Không thể khởi tạo model: {e}", file=sys.stderr)
    sys.exit(1)

PATCH_SIZE = 512 # Kích thước của mỗi patch

# Đọc từng dòng (đường dẫn ảnh HDFS) từ Standard Input
for hdfs_image_path in sys.stdin:
    hdfs_image_path = hdfs_image_path.strip()
    if not hdfs_image_path:
        continue

    try:
        image_id = os.path.splitext(os.path.basename(hdfs_image_path))[0]
        
        # --- ĐỌC DỮ LIỆU ẢNH TỪ HDFS ---
        # Gọi lệnh hdfs dfs -cat để đọc nội dung file ảnh vào bộ nhớ
        process = subprocess.Popen(
            ["hdfs", "dfs", "-cat", hdfs_image_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            print(f"!!! LỖI MAPPER: Không thể đọc file HDFS '{hdfs_image_path}'. Lỗi: {stderr.decode()}", file=sys.stderr)
            continue
        
        # Đọc dữ liệu bytes vào đối tượng PIL Image
        image_bytes = io.BytesIO(stdout)
        image = Image.open(image_bytes).convert("RGB")
        
        # --- LOGIC CẮT ẢNH THÀNH CÁC PATCH 512x512 ---
        width, height = image.size
        patches_processed = 0
        
        # Duyệt qua ảnh theo từng bước bằng kích thước patch
        for i in range(0, width, PATCH_SIZE):
            for j in range(0, height, PATCH_SIZE):
                left = i
                top = j
                right = min(i + PATCH_SIZE, width)
                bottom = min(j + PATCH_SIZE, height)
                
                # Bỏ qua các patch quá nhỏ ở rìa ảnh để tránh nhiễu
                if (right - left) < PATCH_SIZE / 2 or (bottom - top) < PATCH_SIZE / 2:
                    continue

                patch = image.crop((left, top, right, bottom))
                
                feat_vector = extractor.extract_features_from_tile(patch)
                
                if feat_vector is not None:
                    # Chuyển vector thành chuỗi và in ra output: key <TAB> value
                    feat_str = ",".join([str(f) for f in feat_vector.tolist()])
                    print(f"{image_id}\t{feat_str}")
                    patches_processed += 1
        
        if patches_processed == 0:
            print(f"Cảnh báo: Không xử lý được patch nào cho ảnh {hdfs_image_path}", file=sys.stderr)

    except Exception as e:
        print(f"!!! LỖI MAPPER: Lỗi nghiêm trọng khi xử lý ảnh {hdfs_image_path}: {e}", file=sys.stderr)
        continue

