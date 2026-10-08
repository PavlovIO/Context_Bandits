from data_loading import load_movielens
from features import make_movie_features_movie_lens
from bandits import *
from bandit_environment import SyntheticEnvironment
from experiments import *
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "MovieLensData"

def run_synthetic_experiment():
    K_arms = 30
    d = 30
    num_steps = 5000
    num_runs = 30 
    
    scenarios = ['strong', 'weak', 'noise']
    
    for scenario in scenarios:
        print(f"Сценарий: {scenario}")
        
        # Словарь для хранения статистики по всем запускам
        all_results = {
            'UCB1': [], 'TS': [], 'LinUCB': [], 'LinTS': []
        }

        for run in range(num_runs):
            print(f"  Запуск {run + 1}/{num_runs}")


            env = SyntheticEnvironment(K_arms=K_arms, d=d, scenario=scenario, seed=run)

            policies = {
                'UCB1': ucb1_policy(K_arms=K_arms, c=np.sqrt(2)/5, seed=run),
                'TS': ts_policy(K_arms=K_arms, seed=run),
                'LinUCB': linucb_policy(K_arms=K_arms, d=d, alpha=0.5, seed=run),
                'LinTS': lints_policy(K_arms=K_arms, d=d, v=0.15, seed=run)
            }
            
            for algo_name, policy in policies.items():
                policy.reset()

                mab = MAB(policy=policy, K_arms=K_arms, D_ctx=d)
                
                for step in range(num_steps):
                    context = env.get_context()
                    arm_id = mab.choose(context)
                    reward, best_arm_id, best_reward = env.get_reward_and_optimal(arm_id, context)
                    mab.update(arm_id, reward, best_arm_id, best_reward, context)
                
                all_results[algo_name].append(mab.get_statistics())

        save_results(
            results=all_results,
            filepath=f'results/synth_{scenario}_5000_20.pkl',
        )

        plot_results(all_results, scenario_name=scenario, save_dir="coursework_plots", window=500)

def run_movielens_replay_all_policies(max_epochs: int = 5, num_runs: int = 5, seed: int = 42):
    movies, ratings, users = load_movielens(str(DATA_DIR))
    movies_feat, users_feat = make_movie_features_movie_lens(movies, users)

    top_k_movies = 30
    top_movies = ratings['MovieID'].value_counts().head(top_k_movies).index.tolist()
    
    ratings_data = ratings[ratings['MovieID'].isin(top_movies)].copy()
    ratings_data['Reward'] = (ratings_data['Rating'] >= 5).astype(int)
    
    users_feat.index = users['UserID']
    d_ctx = users_feat.shape[1]
    
    baseline_ctr = ratings_data['Reward'].mean()
    print(f"Baseline CTR (случайная политика): {baseline_ctr:.3f}")
    print(f"Всего строк в логе: {len(ratings_data)}")
    
    all_results = {
        'UCB1': [], 
        'TS': [], 
        'LinUCB': [],
        'LinTS': []
    }
    
    print(f"\nЗапуск Replay с Recycling ({num_runs} запусков, до {max_epochs} эпох)")
    
    for run in range(num_runs):

        print(f"Запуск {run + 1}/{num_runs}")

        
        current_seed = seed + run
        
        # Создаем политики для текущего запуска
        policies = {
            'UCB1': ucb1_policy(K_arms=top_k_movies, c=np.sqrt(2), seed=current_seed),
            'TS': ts_policy(K_arms=top_k_movies, seed=current_seed),
            'LinUCB': linucb_policy(K_arms=top_k_movies, d=d_ctx, alpha=0.5, seed=current_seed),
            'LinTS': lints_policy(K_arms=top_k_movies, d=d_ctx, v=0.15, seed=current_seed)
        }
        
        # Запускаем каждую политику через унифицированную функцию
        for algo_name, policy in policies.items():
            stats = run_single_replay_experiment(
                policy=policy,
                algo_name=algo_name,
                ratings_data=ratings_data,
                users_feat=users_feat,
                max_epochs=max_epochs,
                seed=current_seed,
                verbose=True
            )
            all_results[algo_name].append(stats)
    
    return all_results, baseline_ctr



if __name__ == '__main__':

    run_synthetic_experiment()

    ml1m_results, baseline_ctr = run_movielens_replay_all_policies()
    plot_replay_results(ml1m_results, baseline_ctr)




    