import pandas as pd
import os
import zlib
import csv
from tqdm import tqdm

# ================= 配置区域 =================
INPUT_FILE = 'data/raw/train_data.csv'       # 输入文件路径
OUTPUT_FILE = 'data/raw/train_data_extra.csv' # 输出文件路径
SAMPLE_PERCENT = 25                 # 采样比例
CHUNK_SIZE = 100000                 # 每次处理行数
# ===========================================

def get_separator(filepath, sample_bytes=4096):
    """自动侦测分隔符"""
    sniffer = csv.Sniffer()
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            sample = f.read(sample_bytes)
            # 优先检测常见分隔符
            dialect = sniffer.sniff(sample, delimiters=[',', '|', '\t', ';', ' '])
            print(f" 自动检测到分隔符: [{dialect.delimiter!r}]")
            return dialect.delimiter
    except:
        print("无法检测分隔符，默认使用 '|'")
        return '|'

def is_selected_user(user_id, percent):
    """一致性哈希采样逻辑"""
    uid_str = str(user_id)
    hash_val = zlib.crc32(uid_str.encode('utf-8'))
    return (hash_val % 100) < percent

def process_data_robust():
    if not os.path.exists(INPUT_FILE):
        print(f" 找不到输入文件: {INPUT_FILE}")
        return

    # 1. 自动获取分隔符
    sep = get_separator(INPUT_FILE)
    
    # 2. 获取文件大小
    total_size = os.path.getsize(INPUT_FILE)
    print(f" 开始处理，文件大小: {total_size / (1024**3):.2f} GB")

    # 3. 初始化读取器 (使用 python 引擎容错性更好)
    try:
        reader = pd.read_csv(INPUT_FILE, sep=sep, chunksize=CHUNK_SIZE, 
                             header=0, dtype=str, engine='python', on_bad_lines='skip')
    except Exception as e:
        print(f" 初始化读取失败: {e}")
        return

    first_chunk = True
    mode = 'w'
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with tqdm(total=total_size, unit='B', unit_scale=True, desc="清洗进度") as pbar:
        for chunk in reader:
            # 更新进度条
            pbar.update(chunk.memory_usage(deep=True).sum())

            # --- 智能清洗逻辑 ---
            
            # A. 判断是否需要删除第一列 (防报错机制)
            # 逻辑：如果列数 > 24 (你的标准列数)，通常意味着有多余列
            # 或者如果你明确知道大文件有脏数据，可以保留原有逻辑，但加上长度检查
            if chunk.shape[1] > 24:
                # 认为是“大文件”格式，第一列是无效时间，删除
                chunk_cleaned = chunk.iloc[:, 1:]
            else:
                # 认为是“标准/小文件”格式，列数正常，不删
                chunk_cleaned = chunk

            # B. 再次检查：防止删完没数据了
            if chunk_cleaned.empty or chunk_cleaned.shape[1] == 0:
                continue

            # C. 确定用户ID列 (默认处理后的第0列)
            user_id_col = chunk_cleaned.columns[0]

            # D. 哈希采样
            try:
                mask = chunk_cleaned[user_id_col].apply(lambda x: is_selected_user(x, SAMPLE_PERCENT))
                chunk_kept = chunk_cleaned[mask]
            except Exception as e:
                # 万一 user_id 列找错，防止崩溃
                continue

            # E. 写入文件
            if not chunk_kept.empty:
                # 输出统一用逗号分隔，方便后续 Loader 读取
                chunk_kept.to_csv(OUTPUT_FILE, mode=mode, header=first_chunk, index=False, sep=',')
            
            if first_chunk:
                first_chunk = False
                mode = 'a'

    print(f"\n 处理完成！输出文件: {OUTPUT_FILE}")

if __name__ == "__main__":
    process_data_robust()