"""
================================================================================
腳本名稱：split_yolo_dataset.py
模組定位：YOLO 實例分割資料集統計分層切分與同質性檢驗工具

【核心架構與處理大綱】
一、 檢測常態分佈與同質性之屬性維度（Attributes & Feature Dimensions）
    1. 離散標籤屬性（用於類別分佈同質性與平衡檢驗）：
       * class_id：目標類別標籤（0: primary_crown, 1: trunk_wood, 2: nearby_tree）。
       * class_frequency / instance_ratio：各類別在資料集中的實例計數與佔比。
       * total_instances_per_image：每張影像的標註實例密度（Complexity Density）。
    2. 連續幾何形態屬性（用於常態性檢定與雙樣本同質性 KS 檢定）：
       * area_ratio：多邊形外接框正規化面積佔比（w * h），衡量物體尺度長尾分佈。
       * aspect_ratio：多邊形外接框長寬比（w / max(h, 1e-6)），衡量形狀形態（縱長樹幹 vs 橫展枝冠）。
       * point_count：多邊形頂點數量（len(coords) // 2），衡量輪廓複雜度與分割精細度。

二、 資料讀取與特徵解析（Data Ingestion & Feature Parsing）
    1. 驗證影像與對應 txt 標註檔案之存在性。
    2. 解析 YOLO 標註多邊形座標，計算上述各項幾何與類別屬性。
    3. 動態讀取 classes.txt，建立類別名稱對照表。

三、 多目標統計分層優化切分（Optimal Distribution Stratification）
    1. 設定切分比例（預設 8:2）。
    2. 計算類別佔比總變差（Total Variation）以保證罕見類別平衡。
    3. 計算幾何連續特徵（area_ratio 等）之雙樣本 Kolmogorov-Smirnov（KS）統計量。
    4. 透過多目標搜尋最小化綜合損失函數（Total Variation * 10 + KS Stat），挑選最佳分割組合。

四、 目錄結構構建與檔案複製（Artifact Export）
    1. 建立標準 YOLO 目錄結構（images/train, images/val, labels/train, labels/val）。
    2. 將原始影像與標註複製至對應分割資料夾。
    3. 動態產生 data.yaml 設定檔。

五、 統計檢定與分佈報告輸出（Statistical Validation & Reporting）
    1. 類別佔比對齊報表：驗證各類別在 Train 與 Val 之百分比偏差。
    2. 常態性檢定（Shapiro-Wilk Test）：檢定面積、長寬比等屬性是否符合常態分佈。
    3. 雙樣本分佈一致性檢定（Two-sample KS Test）：檢定 Train 與 Val 的特徵屬性是否來自同一個母體分佈（p > 0.05）。
================================================================================
"""

import shutil
import random
from pathlib import Path
from collections import Counter
import numpy as np
from scipy import stats

def parse_yolo_label(label_path: Path):
    """
    解析單一 YOLO 標註檔案，統計各檢測屬性：
    1. class_counts: 各類別實例計數 (class_id)
    2. areas: 多邊形邊界框面積佔比 (area_ratio)
    3. aspect_ratios: 多邊形邊界框長寬比 (aspect_ratio)
    4. point_counts: 多邊形頂點數 (point_count)
    """
    class_counts = Counter()
    areas = []
    aspect_ratios = []
    point_counts = []
    if not label_path.exists():
        return class_counts, areas, aspect_ratios, point_counts

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            cls_id = int(parts[0])
            class_counts[cls_id] += 1
            coords = np.array([float(x) for x in parts[1:]]).reshape(-1, 2)
            w = float(np.ptp(coords[:, 0]))
            h = float(np.ptp(coords[:, 1]))
            areas.append(w * h)
            aspect_ratios.append(w / max(h, 1e-6))
            point_counts.append(len(coords))

    return class_counts, areas, aspect_ratios, point_counts

def load_class_names(classes_file: Path):
    """讀取 classes.txt，若不存在則使用預設類別字典。"""
    default_names = {0: "primary_crown", 1: "trunk_wood", 2: "nearby_tree"}
    if not classes_file.exists():
        return default_names

    names = {}
    with open(classes_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if ":" in line:
                idx_str, name = line.split(":", 1)
                names[int(idx_str.strip())] = name.strip()
            else:
                names[len(names)] = line
    return names if names else default_names

def find_optimal_distribution_split(valid_data, train_ratio=0.8, max_iter=2000, seed_base=42):
    """
    透過多目標統計優化搜尋最佳分割組合：
    同時最小化類別佔比差異（Total Variation）與連續幾何特徵之 KS 統計量，
    確保 Train 與 Val 資料集具備最高度之母體分佈同質性（Identically Distributed）。
    """
    cached_features = []
    for img_p, lbl_p in valid_data:
        c, a, ar, pc = parse_yolo_label(lbl_p)
        cached_features.append((img_p, lbl_p, c, a, ar, pc))

    all_classes = set()
    for item in cached_features:
        all_classes.update(item[2].keys())
    all_classes = sorted(list(all_classes))

    total_n = len(cached_features)
    n_val = int(round(total_n * (1.0 - train_ratio)))
    indices = list(range(total_n))

    best_score = float("inf")
    best_train_indices = None
    best_val_indices = None

    rng = random.Random(seed_base)

    for _ in range(max_iter):
        shuffled = indices[:]
        rng.shuffle(shuffled)
        val_idx_set = set(shuffled[:n_val])
        train_idx_set = set(shuffled[n_val:])

        t_counts = Counter()
        v_counts = Counter()
        t_areas = []
        v_areas = []

        for idx, item in enumerate(cached_features):
            c, a = item[2], item[3]
            if idx in val_idx_set:
                v_counts.update(c)
                v_areas.extend(a)
            else:
                t_counts.update(c)
                t_areas.extend(a)

        t_total = sum(t_counts.values())
        v_total = sum(v_counts.values())
        if t_total == 0 or v_total == 0:
            continue

        # 屬性 1: 類別佔比總變差 (Total Variation)
        cls_diff = sum(abs((t_counts[c] / t_total) - (v_counts[c] / v_total)) for c in all_classes)

        # 屬性 2: 面積佔比之雙樣本 KS 統計量 (Kolmogorov-Smirnov Statistic)
        ks_stat, _ = stats.ks_2samp(t_areas, v_areas)

        # 綜合損失函數
        score = (cls_diff * 10.0) + ks_stat

        if score < best_score:
            best_score = score
            best_train_indices = train_idx_set
            best_val_indices = val_idx_set

    train_data = [(cached_features[i][0], cached_features[i][1]) for i in best_train_indices]
    val_data = [(cached_features[i][0], cached_features[i][1]) for i in best_val_indices]
    return train_data, val_data

def print_distribution_report(train_data, val_data, class_names):
    """執行各項屬性之統計檢定，輸出 Train 與 Val 分佈一致性檢驗報告。"""
    def aggregate_attributes(data_list):
        cls_counter = Counter()
        all_areas = []
        all_aspect_ratios = []
        all_point_counts = []
        for _, lbl_path in data_list:
            c, a, ar, pc = parse_yolo_label(lbl_path)
            cls_counter.update(c)
            all_areas.extend(a)
            all_aspect_ratios.extend(ar)
            all_point_counts.extend(pc)
        return cls_counter, np.array(all_areas), np.array(all_aspect_ratios), np.array(all_point_counts)

    train_c, train_a, train_ar, train_pc = aggregate_attributes(train_data)
    val_c, val_a, val_ar, val_pc = aggregate_attributes(val_data)
    total_train_inst = sum(train_c.values())
    total_val_inst = sum(val_c.values())

    print("\n" + "=" * 60)
    print("【資料集切分統計與分佈同質性檢定報告】")
    print("=" * 60)
    print(f"樣本總數：Train = {len(train_data)} 張, Val = {len(val_data)} 張")
    print(f"標註實例總數：Train = {total_train_inst} 個, Val = {total_val_inst} 個")

    print("\n[屬性 1] 類別標籤與頻率 (class_id & instance_ratio)：")
    for cls_id, cls_name in sorted(class_names.items()):
        t_cnt = train_c.get(cls_id, 0)
        v_cnt = val_c.get(cls_id, 0)
        t_ratio = (t_cnt / total_train_inst * 100) if total_train_inst > 0 else 0
        v_ratio = (v_cnt / total_val_inst * 100) if total_val_inst > 0 else 0
        print(f"  類別 {cls_id} ({cls_name}):")
        print(f"    Train: {t_cnt:5d} 筆 ({t_ratio:5.2f}%) | Val: {v_cnt:5d} 筆 ({v_ratio:5.2f}%)")

    print("\n[屬性 2] 物件面積佔比 (area_ratio = w * h)：")
    if len(train_a) > 0 and len(val_a) > 0:
        print(f"  Train 均值: {train_a.mean():.6f}, 標準差: {train_a.std():.6f}")
        print(f"  Val   均值: {val_a.mean():.6f}, 標準差: {val_a.std():.6f}")
        # 常態性檢定 (Shapiro-Wilk)
        _, p_norm_a = stats.shapiro(train_a[:min(len(train_a), 500)])
        print(f"  常態性檢定 (Shapiro-Wilk) p-value: {p_norm_a:.4e} -> {'常態分佈' if p_norm_a > 0.05 else '非典型常態（自然長尾分佈）'}")
        # 雙樣本 KS 檢定
        ks_a, p_ks_a = stats.ks_2samp(train_a, val_a)
        print(f"  Train vs Val 同質性 (KS Test) p-value: {p_ks_a:.4f} (KS 統計量: {ks_a:.4f}) -> {'高度同質 (p > 0.05)' if p_ks_a > 0.05 else '存在顯著差異'}")

    print("\n[屬性 3] 物件長寬比 (aspect_ratio = w / h)：")
    if len(train_ar) > 0 and len(val_ar) > 0:
        print(f"  Train 均值: {train_ar.mean():.4f}, 標準差: {train_ar.std():.4f}")
        print(f"  Val   均值: {val_ar.mean():.4f}, 標準差: {val_ar.std():.4f}")
        _, p_norm_ar = stats.shapiro(train_ar[:min(len(train_ar), 500)])
        print(f"  常態性檢定 (Shapiro-Wilk) p-value: {p_norm_ar:.4e} -> {'常態分佈' if p_norm_ar > 0.05 else '非典型常態（自然長尾分佈）'}")
        ks_ar, p_ks_ar = stats.ks_2samp(train_ar, val_ar)
        print(f"  Train vs Val 同質性 (KS Test) p-value: {p_ks_ar:.4f} (KS 統計量: {ks_ar:.4f}) -> {'高度同質 (p > 0.05)' if p_ks_ar > 0.05 else '存在顯著差異'}")

    print("\n[屬性 4] 多邊形頂點數 (point_count)：")
    if len(train_pc) > 0 and len(val_pc) > 0:
        print(f"  Train 均值: {train_pc.mean():.2f}, 標準差: {train_pc.std():.2f}")
        print(f"  Val   均值: {val_pc.mean():.2f}, 標準差: {val_pc.std():.2f}")
        ks_pc, p_ks_pc = stats.ks_2samp(train_pc, val_pc)
        print(f"  Train vs Val 同質性 (KS Test) p-value: {p_ks_pc:.4f} (KS 統計量: {ks_pc:.4f}) -> {'高度同質 (p > 0.05)' if p_ks_pc > 0.05 else '存在顯著差異'}")

    print("=" * 60 + "\n")

def main():
    img_dir = Path("data")
    label_dir = Path("samplingAnalysis/output/yolo_labels")
    dataset_dir = Path("dataset")
    classes_file = label_dir / "classes.txt"

    # 1. 建立 YOLO 標準目錄結構
    for split in ["train", "val"]:
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 2. 取得所有影像並配對標註
    valid_exts = {".jpg", ".png", ".jpeg", ".JPG"}
    images = [f for f in img_dir.iterdir() if f.suffix in valid_exts]
    valid_data = []
    for img_path in images:
        label_path = label_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            valid_data.append((img_path, label_path))

    if not valid_data:
        print("未找到有效的標註資料，請確認路徑。")
        return

    # 3. 執行統計最佳化同質切分 (8:2 比例)
    train_data, val_data = find_optimal_distribution_split(valid_data, train_ratio=0.8, max_iter=2000, seed_base=42)

    # 4. 複製檔案至目標資料夾
    def copy_data(data_list, split_name):
        for img_path, label_path in data_list:
            shutil.copy(img_path, dataset_dir / "images" / split_name / img_path.name)
            shutil.copy(label_path, dataset_dir / "labels" / split_name / label_path.name)

    copy_data(train_data, "train")
    copy_data(val_data, "val")

    # 5. 讀取類別並動態生成 data.yaml
    class_names = load_class_names(classes_file)
    names_yaml_lines = [f"  {idx}: {name}" for idx, name in sorted(class_names.items())]
    yaml_content = f"""path: {dataset_dir.absolute()}
train: images/train
val: images/val

names:
""" + "\n".join(names_yaml_lines) + "\n"

    (dataset_dir / "data.yaml").write_text(yaml_content, encoding="utf-8")

    # 6. 輸出各屬性分佈檢定報告
    print_distribution_report(train_data, val_data, class_names)
    print(f"資料集建立完成。Train: {len(train_data)} 筆, Val: {len(val_data)} 筆")

if __name__ == "__main__":
    main()
