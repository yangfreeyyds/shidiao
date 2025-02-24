#词云生成
#freeyyds

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import numpy as np

def generate_wordcloud(freq_file='词频数据.csv',
                      mask_path='img_1.png',
                      output_file='wordcloud.png'):
    # 读取词频数据
    df = pd.read_csv(freq_file)
    word_freq = dict(zip(df['关键词'], df['出现次数']))

    # 解决乱码问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    try:
        mask = np.array(Image.open(mask_path))
    except:
        mask = None

    wc = WordCloud(
        font_path='msyh.ttc',
        mask=mask,
        background_color='white',
        max_words=1000,
        colormap='viridis'
    ).generate_from_frequencies(word_freq)

    plt.imshow(wc)
    plt.axis("off")
    plt.savefig(output_file, dpi=300)
    print(f"词云图已保存至 {output_file}")

if __name__ == '__main__':
    generate_wordcloud()
