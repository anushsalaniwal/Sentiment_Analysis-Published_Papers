from data_loader import load_data
from preprocessing import preprocess_data
from sentiment_model import SentimentModel
from config import Config
from sklearn.metrics import classification_report

def main():
    data_path = Config.DATA_PATH
    model_path = Config.MODEL_PATH

    data = load_data(data_path)
    data = preprocess_data(data)

    sentiment_model = SentimentModel()
    sentiment_model.load(model_path)

    predictions = sentiment_model.model.predict(data['review'])
    predictions = [1 if pred > 0.5 else 0 for pred in predictions]

    print(classification_report(data['label'], predictions))

if __name__ == '__main__':
    main()
