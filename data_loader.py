import json
import pandas as pd

def load_data(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    
    reviews = []
    for paper in data['paper']:
        for review in paper['review']:
            reviews.append({
                'id': review['id'],
                'confidence': review['confidence'],
                'evaluation': review['evaluation'],
                'text': review['text']
            })
    
    df = pd.DataFrame(reviews)
    return df
