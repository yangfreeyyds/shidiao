import logging
import gensim
from gensim import corpora, models
from snownlp import SnowNLP


def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return lines


def preprocess_data(data):
    # 使用 SnowNLP 分词
    processed_data = []
    for text in data:
        text = text.strip()
        if not text:
            continue
        words = SnowNLP(text).words
        processed_data.append(words)
    return processed_data


def perform_lda_analysis(data, num_topics=5):
    # Create a dictionary representation of the documents.
    dictionary = corpora.Dictionary(data)

    # Convert the documents into a document-term matrix.
    corpus = [dictionary.doc2bow(text) for text in data]

    # Apply LDA model
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, random_state=42)

    # Display the topics
    topics = lda_model.print_topics()
    for topic in topics:
        print(topic)


def main():
    # Load your data
    original_data = load_data("original_text.txt")

    if not original_data:
        print("文本数据为空。请检查文件内容。")
        return

    # Preprocess data
    processed_data = preprocess_data(original_data)

    # Perform LDA analysis
    perform_lda_analysis(processed_data, num_topics=5)


if __name__ == '__main__':
    # Optional: Setup logging to Jupyter Console
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    main()
