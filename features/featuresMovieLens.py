import pandas as pd
import numpy as np

def make_movie_features_movie_lens(movies: pd.DataFrame, 
                                   users: pd.DataFrame
                                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Предобработка признаков для датасета MovieLense 1M

    Args:
        movies: Dataframe с фильмами, колонки "MovieID","Title","Genres"
        users: Dataframe с пользователями, колонки "UserID", "Gender", "Age", "Occupation","Zip-code"
    
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (movies, users)
    """
    target_genres = [
        'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
        'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
        'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
    ]
    users = users.copy()
    movies = movies.copy()

    users['Gender'] = (
        users['Gender']
        .fillna('M')
        .str.strip()
        .str.upper()
    )
    gender_dummies = pd.get_dummies(users['Gender'], prefix='Gender', dtype=int)
    users = pd.concat([users, gender_dummies], axis=1)
    users.drop(columns=['Gender', 'Zip-code'], inplace=True)


    age = users['Age'].fillna(users['Age'].median()).astype(int)
    conditions = [
        age < 20,
        age.between(20, 29),
        age.between(30, 39),
        age.between(40, 49),
        age.between(50, 59),
        age >= 60
    ]
    choices = ['<20', '20-29', '30-39', '40-49', '50-69', '60+']
    users['Age_group'] = np.select(conditions, choices, default='18-29')
    age_dummies = pd.get_dummies(users['Age_group'], prefix='Age', dtype=int)  
    users = pd.concat([users.drop(columns=['Age', 'Age_group']), age_dummies], axis=1)

    users = pd.get_dummies(users, columns=['Occupation'], prefix='Occ', dtype=int)


    mask = movies['Title'].str.contains(r'\(\d{4}\)\s*$', na=False)
    movies.loc[mask, 'Year'] = movies.loc[mask, 'Title'].str.extract(r'\((\d{4})\)', expand=False)
    movies.loc[mask, 'Title'] = movies.loc[mask, 'Title'].str.replace(r'\s*\(\d{4}\)\s*$', '', regex=True).str.strip()

    movies['Year'] = pd.to_numeric(movies['Year'], errors='coerce')
    movies['Year_norm'] = ((movies['Year'] - movies['Year'].mean()) / movies['Year'].std()).fillna(0.0)

    clean_genres = (
        movies['Genres']
        .fillna('')
        .str.replace(r'[()]', '', regex=True)
        .str.replace(r'\s*\|\s*', '|', regex=True)
        .str.strip()
    )
    genre_dummies = clean_genres.str.get_dummies(sep='|').reindex(columns=target_genres, fill_value=0)
    movies = pd.concat([movies[['Title', 'Year_norm', 'Genres']], genre_dummies], axis=1).drop(columns=['Genres'])

    movies.drop(columns=['Title'], inplace=True, errors='ignore')
    users.drop(columns=['UserID'], inplace=True)
    users = users.reset_index(drop=True)
    movies = movies.reset_index(drop=True)

    return (movies, users)