# Sistema_II

#  Music Genre Classifier (XGBoost + Lyrics + Features)

Этот проект — попытка классифицировать жанры музыки с помощью комбинации текстов песен и числовых аудио-признаков. Я использовал датасет с Kaggle: ["Music Dataset (1950 - 2019)"](https://www.kaggle.com/datasets/saurabhshahane/music-dataset-1950-to-2019), в котором собрана информация о песнях за несколько десятилетий.

##  Что делает модель

Модель на вход получает:
- Числовые аудио-признаки (`danceability`, `energy`, `valence`, и др.)
- Тексты песен (`lyrics`), преобразованные в TF-IDF
- На выходе — предсказанный жанр (например, rock, pop, jazz, hip hop и т.д.)

Используется XGBoost-классификатор, потому что он хорошо справляется с табличными данными и быстро обучается даже на больших выборках.

##  Что под капотом

- **Язык:** Python 3
- **Модель:** `XGBClassifier` (`xgboost`)
- **Обработка текста:** `TfidfVectorizer` (`sklearn`)
- **Оценка качества:** `classification_report`, accuracy, F1

##  Как запустить

1. Скачай датасет с Kaggle и сохрани как `data.csv`
2. Установи зависимости:
   ```bash
   pip install -r requirements.txt
