import pandas as pd
import numpy as np
from typing import Dict, Any

def run_single_replay_experiment(
    policy,
    algo_name: str,
    ratings_data: pd.DataFrame,
    users_feat: pd.DataFrame,
    max_epochs: int = 10,
    seed: int = 42,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Запускает Replay Method с Epoch Recycling для одной политики.
    
    Args:
        policy: Инициализированный объект политики (UCB1, TS, LinUCB, LinTS)
        algo_name: Название алгоритма
        ratings_data: DataFrame с рейтингами (колонки: UserID, MovieID, Reward)
        users_feat: DataFrame с признаками пользователей (индекс = UserID)
        max_epochs: Максимальное количество эпох для Recycling
        
    Returns:
        Словарь со статистикой:
        {
            'algo_name': str,
            'ctr_history': np.ndarray, - CTR на каждом aligned шаге
            'cumulative_rewards': list, - Накопленные награды по aligned шагам
            'aligned_steps': int, - Всего совпадений с логом
            'epochs_completed': int, - Количество пройденных эпох
            'final_ctr': float, - Финальный CTR
            'total_reward': float - Суммарная награда
        }
    """
    if verbose:
        print(f"Запуск {algo_name}...")
    #сброс алгоритма перед стартом
    policy.reset()
    #загрузка использованных индексов рук
    arms_idx = ratings_data['MovieID'].unique()
    policy.set_arms_idx(arms_idx)
    # Перемешиваем данные
    current_data = ratings_data.sample(frac=1, random_state=seed).reset_index(drop=True)

    aligned_steps = 0
    total_reward = 0.0
    ctr_history = []
    cumulative_rewards = []
    
    epoch = 0
    initial_size = len(current_data)
    
    while len(current_data) > 0 and epoch < max_epochs:
        epoch += 1
        unused_data = []
        
        steps_this_epoch = 0
        matches_this_epoch = 0
        
        for _, row in current_data.iterrows():
            user_id = row['UserID']
            true_movie_id = row['MovieID']
            true_reward = row['Reward']

            context = users_feat.loc[user_id].values.astype(float)

            # l2 - нормализация контекста
            norm = np.linalg.norm(context)
            if norm == 0: norm = 1.0
            context_norm = context / norm

            chosen_arm_id = policy.choose(context_norm)
            
            # Replay: обновляемся только при совпадении
            if chosen_arm_id == true_movie_id:
                reward = float(true_reward)
                policy.update(chosen_arm_id, reward, context_norm)
                
                aligned_steps += 1
                total_reward += reward
                cumulative_rewards.append(total_reward)
                ctr_history.append(total_reward / aligned_steps)
                
                matches_this_epoch += 1
            else:
                # Recycling: сохраняем для следующей эпохи
                unused_data.append(row)
            
            steps_this_epoch += 1
        
        if verbose:
            print(f"    Эпоха {epoch}: {matches_this_epoch}/{steps_this_epoch} совпадений "
                  f"({matches_this_epoch/steps_this_epoch*100:.1f}%), "
                  f"осталось: {len(unused_data)}/{initial_size}")
        
        current_data = pd.DataFrame(unused_data).reset_index(drop=True)
    
    final_ctr = ctr_history[-1] if ctr_history else 0.0
    
    if verbose:
        print(f"    Итого: {aligned_steps} aligned шагов, "
              f"{epoch} эпох, финальный CTR: {final_ctr:.4f}")
    
    return {
        'algo_name': algo_name,
        'ctr_history': np.array(ctr_history),
        'cumulative_rewards': cumulative_rewards,
        'aligned_steps': aligned_steps,
        'epochs_completed': epoch,
        'final_ctr': final_ctr,
        'total_reward': total_reward
    }

