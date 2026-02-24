import pandas as pd
import base64
import numpy as np
import csv

class DataLoader:
    def __init__(self):
        # 标准英文列名
        self.column_names = [
            'user_id', 'target', 'identifier', 'trigger_time', 'page_name', 
            'page_url', 'platform_type', 'area_name', 'sub_area_name',
            'traffic_slot_name', 'sub_traffic_slot_name', 'current_item_name',
            'current_item_price', 'p_code', 'item_price', 'item_name',
            'coupon_name', 'process_step', 'interface_name', 'source_page',
            'source_area', 'cart_count', 'login_type', 'phone_seq'
        ]

    @staticmethod
    def decode_phone(text):
        try:
            text_str = str(text).strip()
            if pd.isna(text_str) or text_str in ['', 'nan', 'None']: return "unknown_user"
            if 'E+' in text_str or 'e+' in text_str:
                text_str = "{:.0f}".format(float(text_str))
            if text_str.endswith('.0'): text_str = text_str[:-2]
            return base64.b64decode(text_str).decode('utf-8')
        except: return text_str

    def load_and_clean(self, file_path):
        df = None
        # 1. 读取文件
        for sep in [',', '\t', '|']:
            try:
                df = pd.read_csv(
                    file_path, sep=sep, engine='python', encoding='utf-8-sig', 
                    on_bad_lines='skip', header=0, names=self.column_names, 
                    dtype=str, quoting=csv.QUOTE_MINIMAL
                )
                if df.shape[1] == 24: 
                    print(f"成功识别分隔符: [{sep}]")
                    break 
            except: continue
        
        if df is None: raise Exception("文件加载失败")
        
        # 2. ID 解密
        if 'user_id' in df.columns:
            df['user_id'] = df['user_id'].apply(self.decode_phone)
        
        # 3. Target 清洗
        if 'target' in df.columns:
            # 这里的逻辑是：如果是 1 则为 1，其他(2, 0, null)都归为 0
            df['target'] = df['target'].apply(lambda x: 1 if str(x).strip().split('.')[0] == '1' else 0)
        
        # 4. 时间清洗
        time_col = next((c for c in df.columns if '时间' in c or 'trigger_time' in c), None)
        if time_col:
            def fix_t(v):
                v_s = str(v).strip()
                try:
                    if 'E+' in v_s or 'e+' in v_s: v_s = "{:.0f}".format(float(v_s))
                    return v_s.split('.')[0]
                except: return v_s
            df['t_fix'] = df[time_col].apply(fix_t)
            df['trigger_time'] = pd.to_datetime(df['t_fix'], format='%Y%m%d%H%M%S', errors='coerce')
            df = df.dropna(subset=['trigger_time'])

        # 5. 数值列清洗
        for c in ['item_price', 'cart_count', 'current_item_price']:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)

        # ==========================================
        # 【关键修复】补全所有中英文映射
        # ==========================================
        col_mapping = {
            'login_type': '登录方式',
            'page_name': '页面名称',
            'item_price': '商品价格',
            'process_step': '办理步骤',
            'cart_count': '加购数量',
            'identifier': '标识符',
            'area_name': '区域名称',
            'interface_name': '接口名称',
            'item_name': '商品名称',
            # ↓↓↓ 刚才缺少的字段都在这里补齐了 ↓↓↓
            'platform_type': '平台类型', 
            'traffic_slot_name': '流量位名称',
            'current_item_name': '当前流量位商品名称',
            'source_page': '来源页面',
            'coupon_name': '使用优惠券名称'
        }
        
        for eng, chn in col_mapping.items():
            if eng in df.columns:
                df[chn] = df[eng] # 复制一列中文的给特征工程用
        
        return df.sort_values(['user_id', 'trigger_time']).reset_index(drop=True)