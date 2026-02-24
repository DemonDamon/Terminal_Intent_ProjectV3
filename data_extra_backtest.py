import pandas as pd
import os

# ================= 配置区域 =================
# 输入：由 data_extra_withdata.py 生成的10月预测数据
INPUT_PREDICT_DATA = 'data/raw/predict_data_10.csv'

# 输入：回测用户名单文件
INPUT_USER_LIST = 'data/raw/回测用户名单.csv'

# 输出：筛选后的回测数据
OUTPUT_BACKTEST_DATA = 'data/raw/predict_data_10_backtest.csv'
# ===========================================

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

    # 2. 检查10月数据文件是否存在
    if not os.path.exists(INPUT_PREDICT_DATA):
        print(f"[错误] 找不到10月预测数据文件: {INPUT_PREDICT_DATA}")
        print("请先运行 data_extra_withdata.py 生成该文件。")
        return

    print(f" 开始读取10月数据: {INPUT_PREDICT_DATA} ...")
    
    try:
        # 读取10月数据
        # 注意：data_extra_withdata.py 生成的文件第一列是 user_id
        df_oct = pd.read_csv(INPUT_PREDICT_DATA, dtype={'user_id': str}, low_memory=False)
        
        # 确保 user_id 列存在且格式统一
        if 'user_id' not in df_oct.columns:
            print(f"[错误] 10月数据中找不到 'user_id' 列。")
            return
            
        df_oct['user_id'] = df_oct['user_id'].str.strip()
        
        print(f" 原始10月数据行数: {len(df_oct)}")

        # 3. 匹配数据
        # 筛选出 user_id 在 target_users 集合中的行
        mask = df_oct['user_id'].isin(target_users)
        df_backtest = df_oct[mask]
        
        match_count = len(df_backtest)
        matched_user_count = df_backtest['user_id'].nunique()
        
        print(f" 匹配完成！命中行数: {match_count}")
        print(f" 命中用户数: {matched_user_count} / {len(target_users)} (名单总数)")

        # 4. 保存结果
        if match_count > 0:
            os.makedirs(os.path.dirname(OUTPUT_BACKTEST_DATA), exist_ok=True)
            df_backtest.to_csv(OUTPUT_BACKTEST_DATA, index=False, encoding='utf-8')
            print(f" 结果已保存至: {OUTPUT_BACKTEST_DATA}")
        else:
            print(" [警告] 没有匹配到任何数据，未生成输出文件。")
            print(" 可能原因：回测名单的用户不在生成的10月样本数据中。")

    except Exception as e:
        print(f"[错误] 处理数据时发生异常: {e}")

if __name__ == "__main__":
    process_backtest_data()