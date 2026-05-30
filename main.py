from data_loading import load_movielens
from features import make_movie_features_movie_lens
from bandits import LinUCBShared, LinTSShared, UCB1, ThompsonSampling
from BanditEnvironment import BanditEnvironment
from utils import ExperimentsManager, ExperimentConfig
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def main():
    # get datasets and extract features
    movies_df, ratings_df, users_df = load_movielens('.\\MovieLensData')
    movies_f, users_f = make_movie_features_movie_lens(movies_df, users_df) # (n_arms, 19), (n_users, 23)
    target_genres = [
        'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
        'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
        'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
    ]

    user_scaler = StandardScaler()
    arm_scaler = StandardScaler()

    X_user_norm = user_scaler.fit_transform(users_f.values)
    X_arm_norm  = arm_scaler.fit_transform(movies_f.values)

    rng = np.random.default_rng(seed=42)
    d_context = X_arm_norm.shape[1] + X_user_norm.shape[1]
    true_weights = rng.normal(0, 0.05, size=d_context).astype(np.float32)

    env = BanditEnviroment(X_arm_norm, X_user_norm, true_weights, seed=42)

    LinUCBGrid: list[ExperimentConfig] = [
        ExperimentConfig("LinUCB_alpha0.1", LinUCBShared, {"alpha": 0.1}),
        ExperimentConfig("LinUCB_alpha0.2", LinUCBShared, {"alpha": 0.2}),
        ExperimentConfig("LinUCB_alpha0.3", LinUCBShared, {"alpha": 0.3}),
    ]
    LinTSGrid: list[ExperimentConfig] = [
        ExperimentConfig("LinTS_sigma0.05", LinTSShared, {"sigma": 0.05}),
        ExperimentConfig("LinTS_sigma0.075", LinTSShared, {"sigma": 0.075}),
        ExperimentConfig("LinTS_sigma0.1", LinTSShared, {"sigma": 0.1}),
    ]

    manager = ExperimentsManager(env, horizon=5000, n_runs=30, window=1000, seed=42, debug=False)
    results = manager.run_exepriments(experiments=LinUCBGrid)
    manager.plot_results(
        results,
        output_dir="experiments/LinUCBGrid",
        window_size=1000,
        figsize=(11,6)
    )
    results = manager.run_exepriments(experiments=LinTSGrid)
    manager.plot_results(
        results,
        output_dir="experiments/LinTSGrid",
        window_size=1000,
        figsize=(11,6)
    )


if __name__ == "__main__":
    main()