# | 自变量（分组） | 因变量（连续）          | 研究问题                                 |
# |----------------|-------------------------|------------------------------------------|
# | 年龄段(2)      | 整体满意度(13)          | 不同年龄段满意度差异                     |
# | 居住区域(6)     | 功能创新评分(18-1)      | 居住距离对创新感知的影响                 |
# | 学历(3)         | 推荐意愿(15)            | 教育程度与推荐意愿的关系                 |
# | 收入水平(4)     | 形象提升评分(18-2)      | 收入差异对城市形象提升的感知差异         |

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def load_and_preprocess(file_path):
    """数据加载与预处理"""
    df = pd.read_excel(file_path)

    # 变量编码转换（根据实际列名调整）
    df['年龄'] = df['年龄'].map({
        1: '18岁以下', 2: '18~25', 3: '26~30',
        4: '31~40', 5: '41~50', 6: '51~60', 7: '60以上'
    })

    df['居住区域'] = df['居住区域'].map({
        1: '紧邻CBD', 2: '较近核心区',
        3: '主城其他', 4: '远郊县市'
    })

    df['学历'] = df['学历'].map({
        1: '初中及以下', 2: '高中/中专',
        3: '大专/本科', 4: '硕士及以上'
    })

    df['收入水平'] = df['收入水平'].map({
        1: '≤3000', 2: '3001-5000',
        3: '5001-10000', 4: '≥10001'
    })

    return df


def create_composite_score(df):
    """创建综合评分（带索引对齐）"""
    variables = [
        '满意度',  # 满意度
        '创新评分',  # 创新评分
        '推荐意愿',  # 推荐意愿
        '形象评分'  # 形象评分
    ]

    # 记录原始索引
    original_index = df.index

    # 创建临时数据副本
    temp_df = df[variables].copy().dropna()

    # 标准化处理
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(temp_df)

    # 主成分分析（PCA）
    from sklearn.decomposition import PCA
    pca = PCA(n_components=1)
    scores = pca.fit_transform(scaled_data)

    # 创建全空列后填充有效数据
    df['综合评分'] = np.nan
    df.loc[temp_df.index, '综合评分'] = scores.flatten()

    return df


def enhanced_anova(df, group_col, alpha=0.05):
    """增强版单因素方差分析"""
    valid_df = df[[group_col, '综合评分']].dropna()

    # 检查有效分组数量
    group_counts = valid_df[group_col].value_counts()
    valid_groups = group_counts[group_counts >= 5].index.tolist()  # 每组至少5个样本

    if len(valid_groups) < 2:
        print(f"有效分组不足：{group_col} 只有 {len(valid_groups)} 个有效分组")
        return

    filtered_df = valid_df[valid_df[group_col].isin(valid_groups)]

    # 方差齐性检验
    groups = [g['综合评分'].values for _, g in filtered_df.groupby(group_col)]
    levene_stat, levene_p = stats.levene(*groups)

    # 选择分析方法
    if levene_p > alpha:
        f_stat, p_value = stats.f_oneway(*groups)
        method = 'ANOVA'
    else:
        from pingouin import welch_anova
        res = welch_anova(data=filtered_df, dv='综合评分', between=group_col)
        f_stat = res.loc[0, 'F']
        p_value = res.loc[0, 'p-unc']
        method = "Welch's ANOVA"

    # 输出结果
    print(f"\n=== {group_col} 对综合评分的影响分析 ===")
    print(f"有效样本量：{len(filtered_df)}")
    print(f"分析方法：{method}")
    print(f"F统计量：{f_stat:.2f}")
    print(f"P值：{p_value:.4f}")

    # 可视化
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=group_col, y='综合评分', data=filtered_df)
    plt.title(f'{group_col} - 综合评分分布')
    plt.xticks(rotation=45)
    plt.show()

    # 事后检验（当p<0.05时）
    if p_value < alpha and len(valid_groups) > 2:
        print("\n事后检验（Tukey HSD）：")
        tukey = pairwise_tukeyhsd(filtered_df['综合评分'], filtered_df[group_col])
        print(tukey.summary())


def main():
    # 数据加载与预处理
    file_path = "../data/问卷数据.xlsx"
    df = load_and_preprocess(file_path)

    # 创建综合评分
    df = create_composite_score(df)

    # 综合评分分布可视化
    plt.figure(figsize=(8, 5))
    sns.histplot(df['综合评分'], kde=True)
    plt.title('综合评分分布')
    plt.show()

    # 执行单因素方差分析
    demographic_vars = ['年龄', '居住区域', '学历', '收入水平']
    for var in demographic_vars:
        enhanced_anova(df, var)


if __name__ == "__main__":
    main()
