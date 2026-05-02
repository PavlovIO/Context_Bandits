import pandas as pd
import numpy as np


def make_movie_features_movie_lens(movies: pd.DataFrame, 
                                   users: pd.DataFrame, 
                                   ratings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Makes matricies of features from DataFrames for MovieLens
    
    Parameters
    ----------
    movies : pd.DataFrame
    users : pd.DataFrame
    ratings : pd.DataFrame

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        DataFranes with features of **movies, users**

        **movies** : MovieID, Title, Year(str), Genres(one-hot)
        
        **users** : UserID, Age, Occupation, Gender(0F/1M), Genres preferences(mean reward for genre)
    """
    target_genres = [
        'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
        'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
        'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
    ]
    users = users.copy()
    movies = movies.copy()
    ratings = ratings.copy()

    gender_map = {'M': 1, 'F': 0}
    users['Gender'] = (
        users['Gender']
        .fillna('')
        .str.strip()
        .str.upper()
        .map(gender_map)
        .fillna(0)
        .astype(int)
    )
    users.drop(columns=['Gender', 'Zip-code'], inplace=True)

    mask = movies['Title'].str.contains(r'\(\d{4}\)\s*$', na=False)
    movies.loc[mask, 'Year'] = movies.loc[mask, 'Title'].str.extract(r'\((\d{4})\)', expand=False)
    movies.loc[mask, 'Title'] = movies.loc[mask, 'Title'].str.replace(r'\s*\(\d{4}\)\s*$', '', regex=True).str.strip()

    clean_genres = (
        movies['Genres']
        .fillna('')
        .str.replace(r'[()]', '', regex=True)
        .str.replace(r'\s*\|\s*', '|', regex=True)
        .str.strip()
    )
    genre_dummies = clean_genres.str.get_dummies(sep='|').reindex(columns=target_genres, fill_value=0)
    movies = pd.concat([movies, genre_dummies], axis=1).drop(columns=['Genres'])

    ratings['is_positive'] = (ratings['Rating'] >= 4).astype(int)
    merged = ratings.merge(movies, on='MovieID', how='inner')
    merged = merged.reset_index(drop=True)
    merged[target_genres] = merged[target_genres].mul(merged['is_positive'], axis=0)
    user_genre_prefs = merged.groupby('UserID', sort=False)[target_genres].mean().reset_index()
    users = users.merge(user_genre_prefs, on='UserID', how='left')
    users[target_genres] = users[target_genres].fillna(0.0)

    return (movies, users )






