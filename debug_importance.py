import joblib
import pandas as pd
import os
import sys

print(">>> 正在运行修复版特征分析脚本 v2.0 <<<")

# 确保能找到 src 目录
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 1. 路径核对
model_path = 'models/intent_v2.pkl'
features_path = 'models/feature_names.pkl'

if not os.path.exists(model_path) or not os.path.exists(features_path):
    print("错误：模型或特征名单文件不存在，请先跑 train_pipline.py")
else:
    # 2. 加载
    try:
        model_wrapper = joblib.load(model_path)
        feature_names = joblib.load(features_path)
    except Exception as e:
        print(f"文件加载失败: {e}")
        exit()
    
    # 3. 获取内部真实模型
    if hasattr(model_wrapper, 'model'):
        real_model = model_wrapper.model
    else:
        real_model = model_wrapper

    # 4. 兼容性提取特征重要性
    importances = None
    
    try:
        # 情况 A: Sklearn 风格 (LGBMClassifier) - 属性
        if hasattr(real_model, 'feature_importances_'):
            importances = real_model.feature_importances_
            print("提示：检测到 Scikit-Learn 风格模型接口")
            
        # 情况 B: 原生 Booster 风格 - 方法
        # 注意：Booster 对象使用的是方法调用 feature_importance()，而不是属性
        elif hasattr(real_model, 'feature_importance'):
            # importance_type='gain' 表示按贡献度算(更准)
            importances = real_model.feature_importance(importance_type='gain')
            print("提示：检测到原生 Booster 模型接口")
            
        else:
            print("错误：无法从模型对象中提取特征重要性。")
            print(f"当前模型类型: {type(real_model)}")
            print(f"可用属性: {dir(real_model)}") # 打印出来看看有什么
            exit()
            
    except Exception as e:
        print(f"提取特征重要性时发生异常: {e}")
        exit()

    # 5. 长度校验与打印
    print(f"--- 诊断信息 ---")
    print(f"模型记录的特征数: {len(importances)}")
    print(f"名单记录的特征数: {len(feature_names)}")
    
    if len(importances) != len(feature_names):
        print("警告：长度不匹配！说明模型文件和特征名单不是同一次生成的。")
        print("请删除 models/*.pkl 后重新运行训练脚本。")
    else:
        # 6. 生成报表
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values(by='importance', ascending=False)

        # 归一化处理（让数值变成 0~100 的相对分，更容易看）
        if importance_df['importance'].sum() > 0:
            importance_df['importance'] = importance_df['importance'] / importance_df['importance'].sum() * 100

        print("\n=== 模型最看重的特征 Top 20 (按贡献度 %) ===")
        print(importance_df.head(20).to_string(index=False, float_format='%.2f'))
        
        # 保存
        save_file = 'models/feature_importance_rank.csv'
        importance_df.to_csv(save_file, index=False)
        print(f"\n完整特征重要性排行已保存至: {save_file}")