import random
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
    # Combine and label data
    data = [(text, 'positive') for text in pos_data] + [(text, 'negative') for text in neg_data]
    random.shuffle(data)

    kf = KFold(n_splits=k)
    total_pos_accuracy = 0
    total_neg_accuracy = 0

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

        total_pos_accuracy += positive_accuracy
        total_neg_accuracy += negative_accuracy

    # Calculate average accuracies
    average_pos_accuracy = (total_pos_accuracy / k) * 100
    average_neg_accuracy = (total_neg_accuracy / k) * 100
    print(f"Average Positive Test Accuracy: {average_pos_accuracy:.2f}%")
    print(f"Average Negative Test Accuracy: {average_neg_accuracy:.2f}%")
    print(f"Overall Average Test Accuracy: {(average_pos_accuracy + average_neg_accuracy) / 2:.2f}%")


if __name__ == '__main__':
    pos_data = load_data("positive.txt")
    neg_data = load_data("negative.txt")
    k_fold_cross_validation(pos_data, neg_data, k=5)
