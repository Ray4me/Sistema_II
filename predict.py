import pandas as pd
import joblib
from scipy.sparse import hstack

model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
tfidf = joblib.load('tfidf.pkl')
le = joblib.load('label_encoder.pkl')
numerical_features = joblib.load('features.pkl')


input_df = pd.read_csv('input.csv')

missing = set(numerical_features + ['lyrics']) - set(input_df.columns)
if missing:
    raise ValueError(f"❌ В input.csv не хватает столбцов: {missing}")


X_num = input_df[numerical_features]
X_lyrics = input_df['lyrics']


X_num_scaled = scaler.transform(X_num)


X_lyrics_tfidf = tfidf.transform(X_lyrics)


X_combined = hstack([X_num_scaled, X_lyrics_tfidf])


pred = model.predict(X_combined)
pred_genres = le.inverse_transform(pred)


for i, genre in enumerate(pred_genres):
    print(f"🎧 Строка {i+1}: Предсказанный жанр — {genre.upper()}")
