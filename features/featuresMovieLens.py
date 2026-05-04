import pandas as pd
import numpy as np

def make_movie_features_movie_lens(movies: pd.DataFrame, 
                                   users: pd.DataFrame
                                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Makes matricies of demographic and content features from DataLense dataset
    
    Parameters
    ----------
    movies : pd.DataFrame
    users : pd.DataFrame

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        DataFranes with features of **movies, users**

        **movies** : Year_norm, Genres(one-hot)
        
        **users** : Gender(0F/1M), Age(1-7), Occupation(one-hot)
    """
    target_genres = [
        'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
        'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
        'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
    ]
    users = users.copy()
    movies = movies.copy()

    # users
    # one-hot Gender
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
    users.drop(columns=['Zip-code'], inplace=True)

    # obvious ordinal Age
    age_map = {1: 1, 18: 2, 25: 3, 35: 4, 45: 5, 50: 6, 56: 7}
    users['Age'] = (
        users['Age']
        .fillna(3)
        .map(age_map)
        .astype(int)
    )

    # one-hot occupation
    users = pd.get_dummies(users, columns=['Occupation'], prefix='Occ', dtype=int)

    # movies
    # extract separate Title and Year
    mask = movies['Title'].str.contains(r'\(\d{4}\)\s*$', na=False)
    movies.loc[mask, 'Year'] = movies.loc[mask, 'Title'].str.extract(r'\((\d{4})\)', expand=False)
    movies.loc[mask, 'Title'] = movies.loc[mask, 'Title'].str.replace(r'\s*\(\d{4}\)\s*$', '', regex=True).str.strip()
    
    # normalized year
    movies['Year'] = pd.to_numeric(movies['Year'], errors='coerce')
    movies['Year_norm'] = ((movies['Year'] - movies['Year'].mean()) / movies['Year'].std()).fillna(0.0)

    # extract Genre and one-hot encode
    clean_genres = (
        movies['Genres']
        .fillna('')
        .str.replace(r'[()]', '', regex=True)
        .str.replace(r'\s*\|\s*', '|', regex=True)
        .str.strip()
    )
    genre_dummies = clean_genres.str.get_dummies(sep='|').reindex(columns=target_genres, fill_value=0)
    movies = pd.concat([movies[['Title', 'Year_norm','Genres']], genre_dummies], axis=1).drop(columns=['Genres'])

    movies.drop(columns=['Title'], inplace=True, errors='ignore')
    users.drop(columns=['UserID'], inplace=True)
    users = users.reset_index(drop=True)
    movies = movies.reset_index(drop=True)

    return (movies, users)
