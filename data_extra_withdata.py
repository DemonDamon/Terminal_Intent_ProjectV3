import pandas as pd
import os
import zlib
import csv
from tqdm import tqdm

# ================= 配置区域 =================
INPUT_FILE = 'data/raw/train_data.csv'        # 原始大文件路径
TRAIN_OUTPUT = 'data/raw/train_data_89.csv'   # 8-9月训练数据输出路径
PREDICT_OUTPUT = 'data/raw/predict_data_10.csv' # 10月预测数据输出路径

BACKTEST_USER_LIST = 'data/raw/回测用户名单.csv'  # 回测用户名单（10月数据按此名单筛选）

TRAIN_SAMPLEa_PERCENT = 50       # 8-9月数据的采样比例 (50%)
CHUNK_SIZE = 100000             # 每次处理行数
# ===========================================

def get_separator(filepath, sample_bytes=4096):
    """自动侦测分隔符"""
    sniffer = csv.Sniffer()
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            sample = f.read(sample_bytes)
            dialect = sniffer.sniff(sample, delimiters=[',', '|', '\t', ';', ' '])
            print(f" 自动检测到分隔符: [{dialect.delimiter!r}]")
            return dialect.delimiter
    except:
        return '|'

def is_selected_user(user_id, percent):
    """一致性哈希采样"""
    uid_str = str(user_id)
    hash_val = zlib.crc32(uid_str.encode('utf-8'))
    return (hash_val % 100) < percent

def get_month(time_str):
    """从时间字符串解析月份 (例如: 20250826... -> 8)"""
    try:
        s = str(time_str).strip()
        # 假设格式是 20250826... 取第5、6位
        if len(s) >= 6:
            month_str = s[4:6]
            return int(month_str)
    except:
        return -1
    return -1

def load_backtest_users(filepath):
    """读取回测用户名单，返回用户ID集合"""
    if not os.path.exists(filepath):
        print(f"[警告] 找不到回测用户名单: {filepath}，10月数据将跳过提取")
        return None
    try:
        df = pd.read_csv(filepath, sep='|', dtype=str, encoding='utf-8')
        df.columns = [c.strip() for c in df.columns]
        if '号码' not in df.columns:
            print(f"[错误] 在 {filepath} 中找不到 '号码' 列。现有列名: {df.columns.tolist()}")
            return None
        user_ids = set(df['号码'].str.strip().unique())
        print(f" 加载回测用户名单: {len(user_ids)} 个唯一用户")
        return user_ids
    except Exception as e:
        print(f"[错误] 读取用户名单失败: {e}")
        return None

def process_split_data():
    if not os.path.exists(INPUT_FILE):
        print(f" 找不到输入文件: {INPUT_FILE}")
        return

    # 加载回测用户名单（用于筛选10月数据）
    backtest_users = load_backtest_users(BACKTEST_USER_LIST)

    sep = get_separator(INPUT_FILE)
    total_size = os.path.getsize(INPUT_FILE)
    print(f" 开始分流处理，文件大小: {total_size / (1024**3):.2f} GB")

    # 初始化读取器 (不设 header，因为第一列可能是时间，我们手动处理)
    # 假设数据结构: [Col0:Time, Col1:UserID, Col2:Target, ...]
    try:
        reader = pd.read_csv(INPUT_FILE, sep=sep, chunksize=CHUNK_SIZE, 
                             header=None, dtype=str, engine='python', on_bad_lines='skip')
    except Exception as e:
        print(f" 读取失败: {e}")
        return

    # 状态标记
    first_chunk_train = True
    first_chunk_pred = True
    
    # 10月数据计数器
    oct_collected_count = 0
    # 记录已选的10月用户ID，防止同一个人重复被选入预测集
    oct_selected_users = set()

    # 确保输出目录存在
    os.makedirs(os.path.dirname(TRAIN_OUTPUT), exist_ok=True)

    # 这里的列名需要和你之前的 loader.py 保持一致，只是多了一个 Time 列在最前面
    # 之前的列名 (24个): ['user_id', 'target', ...] 
    # 现在的列名 (25个): ['time', 'user_id', 'target', ...]
    output_columns = [
        'user_id', 'target', 'identifier', 'trigger_time', 'page_name', 
        'page_url', 'platform_type', 'area_name', 'sub_area_name',
        'traffic_slot_name', 'sub_traffic_slot_name', 'current_item_name',
        'current_item_price', 'p_code', 'item_price', 'item_name',
        'coupon_name', 'process_step', 'interface_name', 'source_page',
        'source_area', 'cart_count', 'login_type', 'phone_seq'
    ]

    with tqdm(total=total_size, unit='B', unit_scale=True, desc="分流进度") as pbar:
        for chunk in reader:
            pbar.update(chunk.memory_usage(deep=True).sum())
            
            # --- 1. 解析时间列 (第0列) ---
            # 创建临时列 _month 用于筛选
            chunk['_month'] = chunk.iloc[:, 0].apply(get_month)
            
            # --- 2. 提取 8-9 月数据 (训练集) ---
            mask_time_train = chunk['_month'].isin([8, 9])
            if mask_time_train.any():
                df_train = chunk[mask_time_train].copy()
                
                # 哈希采样 50%
                # 注意：假设第1列是 user_id (索引为1)
                user_id_col_idx = 1 
                mask_hash = df_train.iloc[:, user_id_col_idx].apply(lambda x: is_selected_user(x, TRAIN_SAMPLE_PERCENT))
                df_train_final = df_train[mask_hash]

                if not df_train_final.empty:
                    # **关键步骤**：删除第0列(Time)和临时列(_month)，只保留后面24列，对齐之前的格式
                    # iloc[:, 1:-1] 表示从第1列开始取，去掉最后一列(_month)
                    df_to_save = df_train_final.iloc[:, 1:-1]
                    
                    mode = 'w' if first_chunk_train else 'a'
                    # 添加 header 方便后续读取
                    df_to_save.to_csv(TRAIN_OUTPUT, mode=mode, header=output_columns if first_chunk_train else False, index=False)
                    first_chunk_train = False

            # --- 3. 提取 10 月数据 (按回测用户名单筛选，保留全部行为事件) ---
            if backtest_users is not None:
                mask_time_pred = chunk['_month'] == 10
                if mask_time_pred.any():
                    df_pred = chunk[mask_time_pred].copy()

                    # 按用户名单筛选（保留用户的全部行为事件，不做去重）
                    uids = df_pred.iloc[:, 1].astype(str).str.strip()
                    mask_user = uids.isin(backtest_users)
                    df_pred_matched = df_pred[mask_user]

                    if not df_pred_matched.empty:
                        # 统计匹配到的唯一用户数
                        oct_selected_users.update(uids[mask_user].unique())
                        oct_collected_count += len(df_pred_matched)

                        # 同样删除第一列时间 和 最后一列month
                        df_to_save_pred = df_pred_matched.iloc[:, 1:-1]

                        mode = 'w' if first_chunk_pred else 'a'
                        df_to_save_pred.to_csv(PREDICT_OUTPUT, mode=mode, header=output_columns if first_chunk_pred else False, index=False)
                        first_chunk_pred = False
            
            # 内存清理
            del chunk

    print("\n" + "="*40)
    print(f" 处理完成！")
    print(f" 8-9月训练集 (约50%): {TRAIN_OUTPUT}")
    if backtest_users is not None:
        print(f" 10月测试集 (名单匹配 {len(oct_selected_users)}/{len(backtest_users)} 用户, {oct_collected_count} 条事件): {PREDICT_OUTPUT}")
    else:
        print(f" 10月数据: 未提供回测用户名单，已跳过")
    print("="*40)

if __name__ == "__main__":
    process_split_data()