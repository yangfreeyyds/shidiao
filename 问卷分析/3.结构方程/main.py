import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from semopy import Model, Optimizer
from semopy.inspector import inspect
from statsmodels.stats.correlation_tools import cov_nearest
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.stats import zscore
import warnings
import sys
import networkx as nx

# 配置中文显示（在导入seaborn前设置）
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统中文显示
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ----------------------
# 版本检查模块
# ----------------------
def check_versions():
    """检查关键库版本"""
    required = {
        'semopy': '0.4.1',
        'pandas': '1.3.0',
        'statsmodels': '0.13.0'
    }

    issues = []
    for pkg, ver in required.items():
        try:
            current = __import__(pkg).__version__
            if current < ver:
                issues.append(f"{pkg} 需要升级: {current} → {ver}+")
        except ImportError:
            issues.append(f"{pkg} 未安装")

    if issues:
        print("⚠️ 环境问题警告：")
        print("\n".join(issues))
        print("建议执行: pip install --upgrade semopy pandas statsmodels")
        sys.exit(1)


# ----------------------
# 数据预处理模块
# ----------------------
def preprocess_data(df):
    """鲁棒性数据预处理函数"""
    try:
        print("\n=== 预处理开始 ===")
        print(f"输入数据维度: {df.shape}")

        # 步骤1: 处理分类变量
        print("\n[步骤1] 处理分类变量...")
        cat_cols = ['居住区域']  # 根据实际情况调整
        for col in cat_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                print(f"分类变量{col}编码为{len(le.classes_)}类")

        # 步骤2: 处理多选题
        print("\n[步骤2] 处理多选题...")
        q9_cols = ['Q9_央视', 'Q9_社交媒体', 'Q9_朋友', 'Q9_报纸', 'Q9_明星', 'Q9_旅游', 'Q9_其他']
        existing_q9 = [col for col in q9_cols if col in df.columns]
        q9_dummies = df[existing_q9].copy()
        print(f"找到{len(existing_q9)}个多选题列")

        # 步骤3: 处理量表题
        print("\n[步骤3] 计算experience...")
        exp_vars = ['Q12_1', 'Q12_2', 'Q16_1', 'Q16_2']
        valid_exp = [v for v in exp_vars if v in df.columns]
        if len(valid_exp) >= 2:
            df['experience'] = df[valid_exp].mean(axis=1)
        else:
            df['experience'] = np.nan

        # 步骤4: 合并数据
        print("\n[步骤4] 合并数据...")
        base_cols = ['居住区域', '城市规划', 'Q12_3', 'Q11']
        other_cols = ['Q18_1', 'Q18_2', 'Q18_4', 'Q18_5', '满意度', 'Q14', '推荐意愿', '创新评分', '形象评分']
        processed = pd.concat([
            df[[c for c in base_cols if c in df.columns]],
            q9_dummies,
            df[[c for c in other_cols if c in df.columns]],
            df[['experience']]
        ], axis=1)

        # 步骤5: 处理缺失值
        print("\n[步骤5] 缺失值处理...")
        # 删除高缺失率列(>70%)
        missing_rate = processed.isnull().mean()
        high_missing = missing_rate[missing_rate > 0.7].index.tolist()
        if high_missing:
            print(f"移除高缺失列: {high_missing}")
            processed = processed.drop(columns=high_missing)

        # 数值列用中位数填补
        num_cols = processed.select_dtypes(include=np.number).columns
        processed[num_cols] = processed[num_cols].fillna(processed[num_cols].median())

        # 删除剩余缺失行
        processed = processed.dropna()
        print(f"最终数据维度: {processed.shape}")

        # 步骤6: 标准化处理
        print("\n[步骤6] 标准化处理...")
        to_scale = ['满意度', 'Q14', '推荐意愿', 'Q18_1', 'Q18_2']
        scaler = StandardScaler()
        for col in [c for c in to_scale if c in processed.columns]:
            processed[col] = scaler.fit_transform(processed[[col]])

        return processed

    except Exception as e:
        print(f"\n⚠️ 预处理异常: {str(e)}")
        return pd.DataFrame()


# ----------------------
# 数据诊断模块
# ----------------------
class DataDiagnostics:
    """增强型数据质量诊断工具"""

    def __init__(self, data):
        self.data = data
        self.diagnostics = {}

    def run_all_checks(self):
        try:
            self.check_sample_size()
            self.check_missing()
            self.check_correlation()
            self.check_distribution()
            self.check_vif()
            return self.diagnostics
        except Exception as e:
            print(f"诊断异常: {str(e)}")
            return {}

    def check_sample_size(self):
        n, p = self.data.shape
        self.diagnostics['n_p_ratio'] = n / p if p != 0 else 0
        self.diagnostics['sample_comment'] = "充足" if n > 10 * p else "不足"

    def check_missing(self):
        missing = self.data.isnull().sum().max()
        self.diagnostics['missing'] = f"{missing}个" if missing > 0 else "无"

    def check_correlation(self):
        corr = self.data.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        high_corr = upper.stack().index[upper.stack() > 0.9].tolist()
        self.diagnostics['high_corr'] = high_corr or "无"

    def check_distribution(self):
        skewness = self.data.skew().abs()
        self.diagnostics['skewed'] = skewness[skewness > 1].index.tolist() or "无"

    def check_vif(self):
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
            vif = pd.Series(
                [variance_inflation_factor(self.data.values, i)
                 for i in range(self.data.shape[1])],
                index=self.data.columns
            )
            self.diagnostics['high_vif'] = vif[vif > 10].index.tolist() or "无"
        except:
            self.diagnostics['high_vif'] = "计算失败"


import networkx as nx


def plot_sem_graph(params):
    """手工绘制SEM路径图"""
    G = nx.DiGraph()

    # 添加显著路径
    sig = params[params['p-value'] < 0.05]
    for _, row in sig.iterrows():
        if row['op'] == '~':
            G.add_edge(row['rval'], row['lval'],
                       weight=row['Estimate'],
                       label=f"{row['Estimate']:.2f}*")

    # 绘制图形
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)
    edge_labels = {(u, v): d['label'] for u, v, d in G.edges(data=True)}

    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='lightblue')
    nx.draw_networkx_labels(G, pos, font_size=10)
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.title("SEM路径系数图（显著路径）")
    plt.axis('off')
    plt.show()




# ----------------------
# 模型定义
# ----------------------
MODEL_SPEC = """
# 测量模型
城乡融合感知 =~ Q18_1 + Q18_2 + Q12_3 + Q9_社交媒体 
体验质量 =~ experience + 满意度 + Q14
传播效果 =~ Q9_央视 + Q9_社交媒体 + 推荐意愿
可持续发展 =~ 城市规划

# 结构模型
体验质量 ~ 城乡融合感知 + 传播效果
可持续发展 ~ 城乡融合感知 + 体验质量
行为意愿 =~ Q14 + 推荐意愿
行为意愿 ~ 体验质量 + 传播效果
"""

# ----------------------
# 主程序
# ----------------------
if __name__ == "__main__":
    check_versions()  # 先执行环境检查

    # 1. 加载数据
    try:
        raw = pd.read_excel('../data/问卷数据.xlsx')
        print(f"\n原始数据维度: {raw.shape}")
    except Exception as e:
        print(f"数据加载失败: {str(e)}")
        sys.exit(1)

    # 2. 数据预处理
    data = preprocess_data(raw)
    if data.empty:
        print("预处理后数据为空!")
        sys.exit(1)

    # 3. 数据诊断
    diag = DataDiagnostics(data).run_all_checks()
    print("\n=== 数据诊断报告 ===")
    print(f"样本量: {data.shape[0]}, 变量数: {data.shape[1]}")
    print(f"高相关变量对: {diag['high_corr']}")
    print(f"高VIF变量: {diag['high_vif']}")

    # 4. 模型拟合
    try:
        mod = Model(MODEL_SPEC)
        mod.load_dataset(data)
        opt = Optimizer(mod)
        res = opt.optimize('MLW')  # 使用加权最小二乘法


        print("\n=== 模型拟合成功 ===")
        print(inspect(opt))

        params = inspect(opt, what='est', std_est=True)

        # 调用函数
        plot_sem_graph(params)

    except Exception as e:
        print(f"\n模型拟合失败: {str(e)}")
        sys.exit(1)
