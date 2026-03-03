import pandas as pd
import os
import csv
import zlib
from tqdm import tqdm

# ================= 配置区域 =================
# 输入：原始全量数据（回测名单针对全量10月，不能用抽样后的 predict_data_10.csv）
INPUT_RAW_DATA = 'data/raw/train_data.csv'

# 输入：回测用户名单文件
INPUT_USER_LIST = 'data/raw/回测用户名单.csv'

# 输出：筛选后的回测数据
OUTPUT_BACKTEST_DATA = 'data/raw/predict_data_10_backtest.csv'

CHUNK_SIZE = 100000
# ===========================================

# 与 withdata 一致的列名（去掉第0列时间后的24列）
OUTPUT_COLUMNS = [
    'user_id', 'target', 'identifier', 'trigger_time', 'page_name',
    'page_url', 'platform_type', 'area_name', 'sub_area_name',
    'traffic_slot_name', 'sub_traffic_slot_name', 'current_item_name',
    'current_item_price', 'p_code', 'item_price', 'item_name',
    'coupon_name', 'process_step', 'interface_name', 'source_page',
    'source_area', 'cart_count', 'login_type', 'phone_seq'
]

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

def get_month(time_str):
    """从时间字符串解析月份 (例如: 20250826... -> 8)"""
    try:
        s = str(time_str).strip()
        if len(s) >= 6:
            month_str = s[4:6]
            return int(month_str)
    except:
        return -1
    return -1

def load_target_users(filepath):
    """读取回测用户名单，返回用户ID的集合"""
    if not os.path.exists(filepath):
        print(f"[错误] 找不到用户名单文件: {filepath}")
        return set()

    try:
        # 根据文件内容：分隔符为 '|'
        # 格式：浏览时间|号码
        df = pd.read_csv(filepath, sep='|', dtype=str, encoding='utf-8')

        # 清理列名空格 (防止 '号码 ' 或 ' 号码' 的情况)
        df.columns = [c.strip() for c in df.columns]

        if '号码' not in df.columns:
            print(f"[错误] 在 {filepath} 中找不到 '号码' 列。现有列名: {df.columns.tolist()}")
            return set()

        # 提取用户ID并去重，去除首尾空格
        user_ids = set(df['号码'].str.strip().unique())
        print(f" 成功加载回测用户名单，共 {len(user_ids)} 个唯一用户。")
        return user_ids

    except Exception as e:
        print(f"[错误] 读取用户名单失败: {e}")
        return set()

def process_backtest_data():
    # 1. 加载目标用户ID
    target_users = load_target_users(INPUT_USER_LIST)
    if not target_users:
        return

    # 2. 检查原始数据文件是否存在
    if not os.path.exists(INPUT_RAW_DATA):
        print(f"[错误] 找不到原始数据文件: {INPUT_RAW_DATA}")
        return

    sep = get_separator(INPUT_RAW_DATA)
    total_size = os.path.getsize(INPUT_RAW_DATA)
    print(f" 开始从全量数据中筛选10月回测数据，文件大小: {total_size / (1024**3):.2f} GB")

    try:
        # 分块读取原始数据（无表头，与 withdata 一致）
        reader = pd.read_csv(INPUT_RAW_DATA, sep=sep, chunksize=CHUNK_SIZE,
                             header=None, dtype=str, engine='python', on_bad_lines='skip')
    except Exception as e:
        print(f"[错误] 读取失败: {e}")
        return

    first_chunk = True
    total_matched_rows = 0
    matched_users = set()

    with tqdm(total=total_size, unit='B', unit_scale=True, desc="回测筛选进度") as pbar:
        for chunk in reader:
            pbar.update(chunk.memory_usage(deep=True).sum())

            # 解析月份（第0列为时间）
            chunk['_month'] = chunk.iloc[:, 0].apply(get_month)

            # 只保留10月数据
            mask_oct = chunk['_month'] == 10
            if not mask_oct.any():
                del chunk
                continue

            df_oct = chunk[mask_oct].copy()

            # 第1列为 user_id，匹配回测用户名单
            user_id_col_idx = 1
            df_oct['_uid'] = df_oct.iloc[:, user_id_col_idx].str.strip()
            mask_user = df_oct['_uid'].isin(target_users)
            df_matched = df_oct[mask_user]

            if not df_matched.empty:
                # 记录匹配的用户
                matched_users.update(df_matched['_uid'].unique())
                total_matched_rows += len(df_matched)

                # 去掉第0列(时间)和临时列(_month, _uid)，只保留中间24列
                # iloc[:, 1:-2] 即第1列到倒数第3列（去掉最后的 _month 和 _uid）
                df_to_save = df_matched.iloc[:, 1:-2]

                mode = 'w' if first_chunk else 'a'
                df_to_save.to_csv(OUTPUT_BACKTEST_DATA, mode=mode,
                                  header=OUTPUT_COLUMNS if first_chunk else False,
                                  index=False)
                first_chunk = False

            del chunk

    print("\n" + "=" * 40)
    print(f" 处理完成！")
    print(f" 匹配命中行数: {total_matched_rows}")
    print(f" 命中用户数: {len(matched_users)} / {len(target_users)} (名单总数)")
    if total_matched_rows > 0:
        print(f" 结果已保存至: {OUTPUT_BACKTEST_DATA}")
    else:
        print(" [警告] 没有匹配到任何数据，未生成输出文件。")
        print(" 可能原因：回测名单中的用户在原始数据的10月部分中不存在。")
    print("=" * 40)

if __name__ == "__main__":
    process_backtest_data()
