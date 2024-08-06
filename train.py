from data_loader import load_data
from preprocessing import preprocess_data
from sentiment_model import SentimentModel
from sklearn.model_selection import train_test_split

def main():
    data_path = '../data/reviews.json'
    data = load_data(data_path)
    
    print(f"DataFrame head:\n{data.head()}")
    print(f"Columns in the DataFrame before preprocessing: {data.columns}")

    data = preprocess_data(data)
    print(f"Columns in the DataFrame after preprocessing: {data.columns}")

    texts = data['text'].values
    labels = data['evaluation'].astype(int).values

    train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)
    
    sentiment_model = SentimentModel()
    sentiment_model.train(train_texts, train_labels)
    loss, accuracy = sentiment_model.evaluate(test_texts, test_labels)
    
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
