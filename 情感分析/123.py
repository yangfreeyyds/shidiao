import random
import matplotlib.pyplot as plt
import seaborn as sns
from snownlp import sentiment
from snownlp import SnowNLP
from sklearn.model_selection import KFold


def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return lines


def evaluate_model(test_data, expected_label):
    correct_predictions = 0
    for text in test_data:
        sentiment_score = SnowNLP(text.strip()).sentiments
        predicted_label = 'positive' if sentiment_score > 0.5 else 'negative'
        if predicted_label == expected_label:
            correct_predictions += 1
    accuracy = correct_predictions / len(test_data)
    return accuracy


def k_fold_cross_validation(pos_data, neg_data, k=5):
    data = [(text, 'positive') for text in pos_data] + [(text, 'negative') for text in neg_data]
    random.shuffle(data)

    kf = KFold(n_splits=k)
    fold_accuracies = []

    for train_index, test_index in kf.split(data):
        train_data = [data[i] for i in train_index]
        test_data = [data[i] for i in test_index]

        # Separate train_data into positive and negative files for training
        with open("positive_train.txt", 'w', encoding='utf-8') as pos_train_file, open("negative_train.txt", 'w',
                                                                                       encoding='utf-8') as neg_train_file:
            for line, label in train_data:
                if label == 'positive':
                    pos_train_file.write(line)
                else:
                    neg_train_file.write(line)

        # Train the model
        sentiment.train("negative_train.txt", "positive_train.txt")
        sentiment.save("sentiment.marshal")

        # Evaluate the model on the test sets
        sentiment.load("sentiment.marshal")
        pos_test = [line for line, label in test_data if label == 'positive']
        neg_test = [line for line, label in test_data if label == 'negative']

        positive_accuracy = evaluate_model(pos_test, 'positive')
        negative_accuracy = evaluate_model(neg_test, 'negative')

        fold_accuracies.append((positive_accuracy, negative_accuracy))

    return fold_accuracies


def plot_accuracies(accuracies):
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    df_accuracies = {
        "Fold": [],
        "Accuracy": [],
        "Label": []
    }

    for i, (pos_acc, neg_acc) in enumerate(accuracies):
        df_accuracies["Fold"].append(f"Fold {i + 1}")
        df_accuracies["Accuracy"].append(pos_acc * 100)
        df_accuracies["Label"].append("Positive")

        df_accuracies["Fold"].append(f"Fold {i + 1}")
        df_accuracies["Accuracy"].append(neg_acc * 100)
        df_accuracies["Label"].append("Negative")

    sns.barplot(x="Fold", y="Accuracy", hue="Label", data=df_accuracies)
    plt.title("K-Fold Cross-validation Accuracies")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.xticks(rotation=45)
    plt.legend(title="Sentiment")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    pos_data = load_data("positive.txt")
    neg_data = load_data("negative.txt")
    accuracies = k_fold_cross_validation(pos_data, neg_data, k=5)
    plot_accuracies(accuracies)
