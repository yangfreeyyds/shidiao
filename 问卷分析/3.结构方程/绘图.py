import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def visualize_sem(params, figsize=(16, 12), node_size=3000):
    """可视化结构方程模型结果"""
    G = nx.DiGraph()

    # 创建节点分层（潜在变量在上，观测变量在下）
    latent_vars = ['城乡融合感知', '体验质量', '传播效果', '可持续发展', '行为意愿']
    observed_vars = [col for col in params['lval'].unique()
                     if col not in latent_vars and not pd.isna(col)]

    # 添加节点并设置属性
    for node in latent_vars:
        G.add_node(node, layer='latent', color='#FFE4B5', shape='s')
    for node in observed_vars:
        G.add_node(node, layer='observed', color='#E0FFFF', shape='o')

    # 添加结构路径
    structural = params[params['op'] == '~']
    for _, row in structural.iterrows():
        color = '#FF4500' if row['Estimate'] < 0 else '#32CD32'  # 红负/绿正
        style = 'solid' if row['p-value'] < 0.05 else 'dashed'  # 实线显著
        width = np.sqrt(abs(row['Estimate'])) * 2  # 线宽反映系数大小

        G.add_edge(row['rval'], row['lval'],
                   weight=abs(row['Estimate']),
                   label=f"{row['Estimate']:.2f}\n(p={row['p-value']:.3f})",
                   color=color, style=style, width=width)

    # 添加测量模型
    measurement = params[params['op'] == '=~']
    for _, row in measurement.iterrows():
        G.add_edge(row['lval'], row['rval'],
                   label=f"{row['Estimate']:.2f}",
                   color='#808080', style='dotted', width=1)  # 灰色虚线

    # 分层布局
    pos = nx.multipartite_layout(G, subset_key="layer", align='horizontal',
                                 scale=3, center=np.array([0, 0]))

    # 调整潜在变量位置
    for node in latent_vars:
        pos[node][1] += 1  # 潜在变量上移

    # 绘图设置
    plt.figure(figsize=figsize)
    ax = plt.gca()

    # 绘制节点
    for node_type in ['latent', 'observed']:
        nodes = [n for n, d in G.nodes(data=True) if d['layer'] == node_type]
        node_color = [G.nodes[n]['color'] for n in nodes]
        nx.draw_networkx_nodes(G, pos, nodelist=nodes,
                               node_shape='s' if node_type == 'latent' else 'o',
                               node_size=node_size, node_color=node_color,
                               edgecolors='k', linewidths=2, ax=ax)

    # 绘制边
    for style in ['solid', 'dashed', 'dotted']:
        edges = [e for e in G.edges(data=True) if e[2]['style'] == style]
        if edges:
            nx.draw_networkx_edges(G, pos, edgelist=[e[:2] for e in edges],
                                   edge_color=[e[2]['color'] for e in edges],
                                   width=[e[2]['width'] for e in edges],
                                   style=style, arrowsize=25, ax=ax)

    # 添加标签
    edge_labels = {(u, v): d['label'] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_labels(G, pos, font_size=12, font_family='SimHei')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                 font_size=9, label_pos=0.6,
                                 rotate=False, ax=ax)

    # 添加图例
    legend_elements = [
        plt.Line2D([0], [0], color='#32CD32', lw=2, label='正向显著 (p<0.05)'),
        plt.Line2D([0], [0], color='#FF4500', lw=2, label='负向显著 (p<0.05)'),
        plt.Line2D([0], [0], color='gray', lw=2, linestyle='dashed',
                   label='不显著路径 (p≥0.05)'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#FFE4B5',
                   markersize=15, label='潜在变量'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#E0FFFF',
                   markersize=15, label='观测变量')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=12)

    plt.title("结构方程模型可视化结果", fontsize=18, pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

data = {
    'lval': ['体验质量', '体验质量', '可持续发展', '可持续发展', '行为意愿', '行为意愿',
            'Q18_1', 'Q18_2', 'Q12_3', 'Q9_社交媒体', 'Q9_社交媒体', 'experience',
            '满意度', 'Q14', 'Q14', 'Q9_央视', '推荐意愿', '推荐意愿', '城市规划', '传播效果',
            '体验质量', '可持续发展', '城乡融合感知', '行为意愿', 'Q12_3', 'Q14', 'Q18_1',
            'Q18_2', 'Q9_央视', 'Q9_社交媒体', 'experience', '城市规划', '推荐意愿', '满意度'],
    'op': ['~', '~', '~', '~', '~', '~', '~', '~', '~', '~', '~', '~',
          '~', '~', '~', '~', '~', '~', '~', '~~', '~~', '~~', '~~', '~~',
          '~~', '~~', '~~', '~~', '~~', '~~', '~~', '~~', '~~', '~~'],
    'rval': ['城乡融合感知', '传播效果', '城乡融合感知', '体验质量', '体验质量', '传播效果',
            '城乡融合感知', '城乡融合感知', '城乡融合感知', '城乡融合感知', '传播效果', '体验质量',
            '体验质量', '体验质量', '行为意愿', '传播效果', '传播效果', '行为意愿', '可持续发展',
            '传播效果', '体验质量', '可持续发展', '城乡融合感知', '行为意愿', 'Q12_3', 'Q14',
            'Q18_1', 'Q18_2', 'Q9_央视', 'Q9_社交媒体', 'experience', '城市规划', '推荐意愿', '满意度'],
    'Estimate': [0.936865, -0.207423, 0.108942, -0.118022, 1.043195, -0.147920,
               1.000000, 1.013495, 0.736292, -0.024349, -0.002937, 1.000000,
               1.005036, -0.102096, 1.000000, 1.000000, -0.004516, 0.953592,
               1.000000, 0.000000, 0.000000, 0.088836, 0.549082, 0.142079,
               0.337586, 0.418109, 0.451110, 0.436391, 0.234463, 0.155214,
               0.191734, 0.108461, 0.379736, 0.504665],
    'p-value': [0.819417, 0.998398, 0.997262, 0.997197, 0.645239, 0.998411,
               np.nan, 0.0, 0.0, 0.921762, 0.999626, np.nan,
               0.0, 0.938273, np.nan, np.nan, 0.999743, 0.334086,
               np.nan, 1.0, 1.0, 0.0, 0.0, 0.400479,
               0.0, 0.005075, 0.0, 0.0, 0.586633, 0.0,
               0.0, 0.0, 0.005139, 0.0]
}

# 创建DataFrame
params = pd.DataFrame(data)
# 调用可视化函数
visualize_sem(params)
