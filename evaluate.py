from data_loader import load_data
from preprocessing import preprocess_data
from sentiment_model import SentimentModel
from sklearn.model_selection import train_test_split

def evaluate_model():
    data_path = '../data/reviews.json'
    data = load_data(data_path)
    
    data = preprocess_data(data)
    
    texts = data['text'].values
    labels = data['evaluation'].astype(int).values

    _, test_texts, _, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)
    
    sentiment_model = SentimentModel()
    sentiment_model.model.load_weights('sentiment_model.h5')

    loss, accuracy = sentiment_model.evaluate(test_texts, test_labels)
    
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    evaluate_model()
