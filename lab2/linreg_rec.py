import pickle
import re
import nltk
import pandas as pd
import sklearn

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

nltk.download("stopwords")
nltk.download("punkt")
nltk.download('punkt_tab')

def books_preprocessing(books: pd.DataFrame) -> pd.DataFrame:
    """Функция для предобработки таблицы Books.scv"""

    for i in books[books["Year-Of-Publication"].map(str).str.match("[^0-9]")]['Book-Title'].index: #убирает сдвиг в данных 
        stroka = books.iloc[i]
        joined_cell = stroka['Book-Title']
        book_author = joined_cell[joined_cell.find(';')+1:-1]
        book_title = joined_cell[:joined_cell.find("\\")]
        year = stroka['Book-Author']
        col_names = list(books)
        for col_ind in range(len(col_names)):
            if col_names[col_ind] == 'ISBN':
                continue
            if col_names[col_ind] == 'Book-Title':
                books.loc[i, col_names[col_ind]] = book_title
            elif col_names[col_ind] =='Book-Author':
                books.loc[i, col_names[col_ind]] = book_author
            elif col_names[col_ind] == 'Year-Of-Publication':
                books.loc[i, col_names[col_ind]] = year
            else:
                books.loc[i, col_names[col_ind]] = books.iloc[i][col_names[col_ind-1]]

    books['Year-Of-Publication'] = books['Year-Of-Publication'].astype(int)
    books = books[books['Year-Of-Publication']<=2024]
    books = books[col_names[:-3]]
    books = books.dropna()

    return books


def ratings_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Функция для предобработки таблицы Ratings.scv
    Целевой переменной в этой задаче будет средний рейтинг книги,
    поэтому в предобработку (помимо прочего) нужно включить:
    1. Замену оценки книги пользователем на среднюю оценку книги всеми пользователями.
    2. Расчет числа оценок для каждой книги (опционально)."""

    avg_ratings = df.groupby('ISBN')['Book-Rating'].mean().reset_index()
    avg_ratings.rename(columns={'Book-Rating': 'Avg-Rating'}, inplace=True)

    df = df.merge(avg_ratings, on='ISBN', how='left')
    

    df.drop(columns=['Book-Rating', 'User-ID'], inplace=True)
    df = df.drop_duplicates(subset=None, keep='first', inplace=False)

    return df


def title_preprocessing(text: str) -> str:
    """Функция для нормализации текстовых данных в стобце Book-Title:
    - токенизация
    - удаление стоп-слов
    - удаление пунктуации
    Опционально можно убрать шаги или добавить дополнительные.
    """
    stop_words = set(stopwords.words("english"))
    tokens = nltk.word_tokenize(text.lower())
    tokens = [word for word in tokens if (word.isalpha() or word.isdigit()) and word not in stop_words]

    return " ".join(tokens)


def modeling(books: pd.DataFrame, ratings: pd.DataFrame) -> None:
    """В этой функции нужно выполнить следующие шаги:
    1. Бинаризовать или представить в виде чисел категориальные столбцы (кроме названий)
    2. Разбить данные на тренировочную и обучающую выборки
    3. Векторизовать подвыборки и создать датафреймы из векторов (размер вектора названия в тестах – 1000)
    4. Сформировать итоговые X_train, X_test, y_train, y_test
    5. Обучить и протестировать SGDRegressor
    6. Подобрать гиперпараметры (при необходимости)
    7. Сохранить модель"""
    books = books_preprocessing(books)
    ratings = ratings_preprocessing(ratings)
    books['Book-Title'] = books['Book-Title'].apply(title_preprocessing)
    books_with_ratings = pd.merge(books, ratings, on='ISBN', how='left')
    books_with_ratings = books_with_ratings.dropna()
    
    lab_enc_author = LabelEncoder()
    lab_enc_publisher = LabelEncoder()

    books_with_ratings['Book-Author'] = lab_enc_author.fit_transform(books_with_ratings['Book-Author'])
    books_with_ratings['Publisher'] = lab_enc_publisher.fit_transform(books_with_ratings['Publisher'])

    books_with_ratings = books_with_ratings.drop(columns=['ISBN'])

    vectorizer = TfidfVectorizer(max_features=1000)
    X_title = vectorizer.fit_transform(books_with_ratings['Book-Title'])

    X_other = books_with_ratings.drop(columns=['Book-Title', 'Avg-Rating'])
    X = pd.concat([pd.DataFrame(X_title.toarray()), X_other.reset_index(drop=True)], axis=1)
    X.columns = X.columns.astype(str)
    y = books_with_ratings['Avg-Rating']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    numerical_columns = ['Year-Of-Publication', 'Book-Author', 'Publisher']  
    X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
    X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])
    print(X_train)

    linreg = SGDRegressor(max_iter=1000, tol=1e-3, alpha=0.001, random_state=42)
    linreg.fit(X_train, y_train)

    y_pred = linreg.predict(X_test)
    print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))

    with open("d:/Университет/pyad-2024/lab2/linreg.pkl", "wb") as file:
        pickle.dump(linreg, file)

books = pd.read_csv('d:/Университет/pyad-2024/lab2/Books.csv')
ratings = pd.read_csv('d:/Университет/pyad-2024/lab2/Ratings.csv')
modeling(books, ratings)