# Функция чтения
import os
import pandas as pd
import numpy as np

def load_movielens(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: 
    '''
    Функция для загрузки датасета MovieLens 1M
    
    Args:
        data_dir (str): Папка с файлами movies.dat, ratings.dat, users.dat
    
    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Датафреймы в последовательности movies, ratings, users
    '''
    
    ratings = pd.read_csv(
            os.path.join(data_dir, "ratings.dat"),
            sep='::',
            names=["UserID","MovieID","Rating","Timestamp"],
            engine='python',
            encoding='latin-1'
        )

    users = pd.read_csv(
            os.path.join(data_dir, "users.dat"),
            sep='::',
            names=["UserID", "Gender", "Age", "Occupation","Zip-code"],
            engine='python',
            encoding='latin-1'
        )

    movies = pd.read_csv(
            os.path.join(data_dir, "movies.dat"),
            sep='::',
            names=["MovieID","Title","Genres"],
            engine='python',
            encoding='latin-1'
        )

    return movies, ratings, users 