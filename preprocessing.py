import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

nltk.download('stopwords')
nltk.download('punkt')

def preprocess_text(text):
    stop_words = set(stopwords.words('spanish'))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word.lower() for word in tokens if word.lower() not in stop_words and word not in string.punctuation]
    return ' '.join(tokens)

def preprocess_data(df):
    if 'text' not in df.columns:
        raise ValueError("DataFrame does not contain a 'text' column")
    
    df['text'] = df['text'].apply(preprocess_text)
    return df
