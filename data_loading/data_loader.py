# Функция чтения
import os
import pandas as pd
import numpy as np

def load_movielens(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''
    Function to load MovieLense Dataset
    
    Parameters
    ----------
    data_dir : str
               Dircetory of dataset with files movies.dat, ratings.dat, users.dat
    
    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
               DataFrames in order **movies, ratings, users**
    '''

    def parse_data_file(filepath: str, column_names: list[str]) -> pd.DataFrame:
        '''
        modified read_csv
        '''
        return pd.read_csv(
            filepath,
            sep='::',
            names=column_names,
            engine='python',
            encoding='latin-1'
        )
    
    ratings = parse_data_file(
        os.path.join(data_dir, "ratings.dat"),
        column_names=["UserID","MovieID","Rating","Timestamp"]
    )

    users = parse_data_file(
        os.path.join(data_dir, "users.dat"),
        column_names=["UserID","Gender","Age","Occupation","Zip-code"]
    )

    movies = parse_data_file(
        os.path.join(data_dir, "movies.dat"),
        column_names=["MovieID","Title","Genres"]
    )

    return movies, ratings, users