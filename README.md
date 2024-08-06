# Sentiment_Analysis-Published_Papers

## Setup

1. **Clone the repository:**

    ```sh
    git clone <repository_url>
    cd sentiment-analysis-project
    ```

2. **Set up a virtual environment:**

    ```sh
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install required packages:**

    ```sh
    pip install pandas nltk tensorflow tensorflow_hub tensorflow_text scikit-learn
    ```

4. **Run the training script:**

    ```sh
    cd src
    python train.py
    ```

5. **Save the trained model:**

    ```python
    # Inside train.py, after training the model
    sentiment_model.model.save_weights('sentiment_model.h5')
    ```

6. **Run the evaluation script:**

    ```sh
    python evaluate.py
    ```

## Project Structure

- `data/`: Contains the dataset.
- `src/`: Contains the source code.
- `src/data_loader.py`: Script to load the data.
- `src/preprocessing.py`: Script to preprocess the data.
- `src/sentiment_model.py`: Script to define the sentiment analysis model.
- `src/train.py`: Script to train the model.
- `src/evaluate.py`: Script to evaluate the trained model.
