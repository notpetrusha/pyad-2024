import pandas as pd
import pickle
from surprise import SVD
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise.model_selection import GridSearchCV
from surprise import accuracy

def ratings_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Функция для предобработки таблицы Ratings.csv"""
    df = df[df['Book-Rating'] > 0]

    ISBN_count_ratings = df['ISBN'].value_counts().reset_index()
    ISBN_count_ratings.columns = ['ISBN', 'count']
    ISBN_drop = list(ISBN_count_ratings[ISBN_count_ratings['count'] <= 1]['ISBN'])

    User_ID_count_ratings = df['User-ID'].value_counts().reset_index()
    User_ID_count_ratings.columns = ['User-ID', 'count']
    User_ID_drop = list(User_ID_count_ratings[User_ID_count_ratings['count'] <= 1]['User-ID'])

    df = df[~df['User-ID'].isin(User_ID_drop)]
    df = df[~df['ISBN'].isin(ISBN_drop)]

    return df

def modeling(ratings: pd.DataFrame) -> None:
    """В этой функции нужно выполнить следующие шаги:
    1. Разбить данные на тренировочную и обучающую выборки
    2. Обучить и протестировать SVD
    3. Подобрать гиперпараметры (при необходимости)
    4. Сохранить модель"""
    
    ratings.rename(columns={'User-ID': 'user_id', 'ISBN': 'item_id', 'Book-Rating': 'rating'}, inplace=True)

    reader = Reader(rating_scale=(1, 10))

    data = Dataset.load_from_df(ratings[['user_id', 'item_id', 'rating']], reader)

    trainset, testset = train_test_split(data, test_size=0.2)

    param_grid = {
    'n_factors': [ 30, 40],
    'n_epochs': [20, 30],
    'lr_all': [0.0025, 0.005],
    'reg_all': [0.01, 0.02]
    }
    gs = GridSearchCV(SVD, param_grid, measures=['mae'], cv=3)
    gs.fit(data)

    print(f"Лучшие параметры: {gs.best_params['mae']}")
    print(f"Лучший MAE: {gs.best_score['mae']}")

    svd = gs.best_estimator['mae']
    svd.fit(trainset)

    predictions = svd.test(testset)
    mae = accuracy.mae(predictions)

    print(f'MAE: {mae}')
    
    with open("svd.pkl", "wb") as file:
        pickle.dump(svd, file)

df = pd.read_csv('Ratings.csv')

df = ratings_preprocessing(df)
modeling(df)
