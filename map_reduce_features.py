import os
import pandas as pd
import numpy as np
import torch
from pyspark import SparkContext, SparkConf



# ---------------- Reduce Function ----------------
def reduce_features(features_list):
    """
    Nhận list các feature arrays [(N1,D), (N2,D), ...]
    → trả về một array (N_total, D) bằng cách ghép dọc (axis=0)
    """
    # Loại bỏ None
    feats = [f for f in features_list if f is not None]
    if len(feats) == 0:
        return None
    if len(feats) == 1:
        return feats[0]  # đã là (N,D)
    
    # Ghép theo hàng
    return np.concatenate(feats, axis=0)


# ---------------- Map Function ----------------
def extract_with_model(x, ckpt_path):
    try:
        from feature_extractor import FeatureExtractor
        slide_id, img_path = x
        extractor = FeatureExtractor(ckpt_path=ckpt_path)
        feat = extractor.extract_features(img_path)
        print(f"✅ Extracted features for {slide_id}")
        return (slide_id, feat)
    except Exception as e:
        print(f"❌ Error processing {x}: {e}")
        return (x[0], None)

# ---------------- Main Pipeline ----------------
def main(csv_path, source_folder, output_csv, features_dir, ckpt_path="ckpts/conch.pth"):
    # Spark config (tăng memory một chút)
    conf = SparkConf().setAppName("FeatureMapReduce") \
                      .set("spark.driver.memory", "6g") \
                      .set("spark.executor.memory", "4g")
    sc = SparkContext(conf=conf)

    # Đọc metadata
    df = pd.read_csv(csv_path)
    slide_ids = df['image_id'].tolist()
    image_paths = [os.path.join(source_folder, f"{sid}.jpg") for sid in slide_ids]
    print(f"📂 Loaded {len(slide_ids)} slide IDs from {csv_path}")

    # Tạo thư mục lưu .pt
    os.makedirs(features_dir, exist_ok=True)

    # Tạo RDD
    rdd = sc.parallelize(zip(slide_ids, image_paths), numSlices=2)

    # Map (load model trong worker)
    mapped = rdd.map(lambda x: extract_with_model(x, ckpt_path))

    # Shuffle & Reduce
    reduced = mapped.groupByKey().mapValues(lambda feats: reduce_features(list(feats)))

    # Collect kết quả
    results = reduced.collect()
    print(f"📊 Collected {len(results)} feature vectors")

    # Lưu CSV
    df_out = pd.DataFrame(results, columns=["image_id", "features"])
    df_out.to_csv(output_csv, index=False)
    print(f"✅ Saved global features to {output_csv}")

    # Lưu từng ảnh thành .pt
    for slide_id, feat in results:
      if feat is not None:
        out_path = os.path.join(features_dir, f"{slide_id}.pt")
        arr = torch.tensor(feat, dtype=torch.float32)   # (N, D)
        print(f"{slide_id}: {arr.shape}")               # debug xem shape
        torch.save(arr, out_path)

    print(f"✅ Saved individual .pt files in {features_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--source_folder", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--features_dir", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, default="ckpts/conch.pth")
    args = parser.parse_args()

    main(
        csv_path=args.csv_path,
        source_folder=args.source_folder,
        output_csv=args.output_csv,
        features_dir=args.features_dir,
        ckpt_path=args.ckpt_path
    )


