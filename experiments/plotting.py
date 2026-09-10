import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any

def compute_rolling_stats(data_2d: np.ndarray, window: int) -> tuple:
    num_runs, T = data_2d.shape
    valid_T = T - window + 1
    
    #вычисляем скользящее среднее для каждого запуска
    rolling_means = np.zeros((num_runs, valid_T))
    for i in range(num_runs):
        rolling_means[i] = np.convolve(data_2d[i], np.ones(window) / window, mode='valid')
    
    #агрегируем по запускам
    mean_across_runs = np.mean(rolling_means, axis=0)
    std_across_runs = np.std(rolling_means, axis=0)
    
    return mean_across_runs, std_across_runs


def plot_results(results: Dict[str, List[Dict[str, Any]]], 
                 scenario_name: str, 
                 save_dir: str = "plots", 
                 window: int = 1000):
    """
    Строит и сохраняет графики метрик для нескольких алгоритмов по множеству запусков.
    
    Args:
        results: Словарь { 'AlgorithmName': [stats_run_1, stats_run_2, ...] }
        scenario_name: Название сценария для заголовков и имен файлов
        save_dir: Папка для сохранения графиков
        window: Размер окна для скользящего среднего
    """
    os.makedirs(save_dir, exist_ok=True)
    
    colors = {'UCB1': '#1f77b4', 'TS': '#ff7f0e', 'LinUCB': '#2ca02c', 'LinTS': '#d62728'}
    linestyles = {'UCB1': '-', 'TS': '--', 'LinUCB': '-', 'LinTS': '--'}
    
    first_algo = list(results.keys())[0]
    T = len(results[first_algo][0]['history_rewards'])
    steps = np.arange(T)
    
    #Кумулятивное сожаление
    plt.figure(figsize=(10, 6))
    
    for algo, runs in results.items():
        cum_regret_2d = np.array([run['cumulative_regret'] for run in runs])
        
        mean_regret = np.mean(cum_regret_2d, axis=0)
        std_regret = np.std(cum_regret_2d, axis=0)
        
        color = colors.get(algo, 'black')
        ls = linestyles.get(algo, '-')
        
        plt.plot(steps, mean_regret, label=algo, color=color, linestyle=ls, linewidth=2)
        plt.fill_between(steps, mean_regret - std_regret, mean_regret + std_regret, 
                         color=color, alpha=0.2)
    
    plt.xlabel('Шаг (t)', fontsize=12)
    plt.ylabel('Кумулятивный регрет', fontsize=12)
    plt.title(f'Кумулятивный регрет: {scenario_name}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path_regret = os.path.join(save_dir, f"cumulative_regret_{scenario_name}.png")
    plt.savefig(save_path_regret, dpi=300, bbox_inches='tight')
    plt.close()

    #Средняя награда
    plt.figure(figsize=(10, 6))
    
    for algo, runs in results.items():
        rewards_2d = np.array([run['history_rewards'] for run in runs])
        
        mean_rew, std_rew = compute_rolling_stats(rewards_2d, window)

        steps_rolling = np.arange(window - 1, T)
        
        color = colors.get(algo, 'black')
        ls = linestyles.get(algo, '-')
        
        plt.plot(steps_rolling, mean_rew, label=algo, color=color, linestyle=ls, linewidth=2)
        plt.fill_between(steps_rolling, mean_rew - std_rew, mean_rew + std_rew, 
                         color=color, alpha=0.2)

    
    plt.xlabel('Шаг (t)', fontsize=12)
    plt.ylabel(f'Средняя награда (окно = {window})', fontsize=12)
    plt.title(f'Сходимость по награде: {scenario_name}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path_rew = os.path.join(save_dir, f"average_reward_{scenario_name}.png")
    plt.savefig(save_path_rew, dpi=300, bbox_inches='tight')
    plt.close()

    #Доля оптимального действия
    plt.figure(figsize=(10, 6))
    
    for algo, runs in results.items():
        opt_act_2d = np.array([ [1 if action == best_action else 0 
            for action, best_action in zip(run['history_actions'], run['history_best_actions'])]
            for run in runs
        ])
        
        mean_opt, std_opt = compute_rolling_stats(opt_act_2d, window)
        steps_rolling = np.arange(window - 1, T)
        
        color = colors.get(algo, 'black')
        ls = linestyles.get(algo, '-')
        
        plt.plot(steps_rolling, mean_opt, label=algo, color=color, linestyle=ls, linewidth=2)
        plt.fill_between(steps_rolling, mean_opt - std_opt, mean_opt + std_opt, 
                         color=color, alpha=0.2)

    
    plt.xlabel('Шаг (t)', fontsize=12)
    plt.ylabel(f'Доля оптимальных действий (окно = {window})', fontsize=12)
    plt.title(f'Частота выбора оптимального действия: {scenario_name}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.0)
    plt.tight_layout()
    
    save_path_opt = os.path.join(save_dir, f"optimal_action_rate_{scenario_name}.png")
    plt.savefig(save_path_opt, dpi=300, bbox_inches='tight')
    plt.close()

def plot_replay_results(results: Dict[str, list],
                        baseline_ctr: float,
                        save_dir: str = "coursework_plots",
                        scenario_name: str = "MovieLens_Replay"):

    """
        Строит и сохраняет графики CTR для нескольких алгоритмов по эксперименту на данных MovieLense.
        
        Args:
            results: Словарь { 'AlgorithmName': [stats_run_1, stats_run_2, ...] }
            baseline_ctr: Базовый CTR при случайно политике
            save_dir: Папка для сохранения графиков
            scenario_name: Название сценария для заголовков и имен файлов
    """
    
    os.makedirs(save_dir, exist_ok=True)
    
    colors = {'UCB1': '#1f77b4', 'TS': '#ff7f0e', 'LinUCB': '#2ca02c', 'LinTS': '#d62728'}
    linestyles = {'UCB1': '-', 'TS': '--', 'LinUCB': '-', 'LinTS': '--'}

    #поиск минимального количество совпадений
    min_len_all = +np.inf
    for algo_name, runs in results.items():
        all_ctr = [run['ctr_history'] for run in runs]
        min_len = min(len(c) for c in all_ctr)
        min_len_all = min(min_len_all, min_len)
    
    plt.figure(figsize=(12, 7))
    
    for algo_name, runs in results.items():

        all_ctr = [run['ctr_history'] for run in runs]
        ctr_matrix = np.array([c[:min_len_all] for c in all_ctr])
        
        mean_ctr = np.mean(ctr_matrix, axis=0)
        std_ctr = np.std(ctr_matrix, axis=0)
        
        color = colors.get(algo_name, 'black')
        ls = linestyles.get(algo_name, '-')
        
        plt.plot(mean_ctr, label=algo_name, color=color, linestyle=ls, linewidth=2)
        plt.fill_between(range(len(mean_ctr)), 
                         mean_ctr - std_ctr, 
                         mean_ctr + std_ctr,
                         color=color, alpha=0.2)
    
    plt.axhline(y=baseline_ctr, color='gray', linestyle=':', linewidth=1.5, 
                label=f'Baseline (random) = {baseline_ctr:.3f}')
    
    plt.xlabel('Aligned Steps', fontsize=12)
    plt.ylabel('Cumulative CTR', fontsize=12)
    plt.ylim((0.2, 0.8))
    plt.title(f'Replay Method с Recycling: {scenario_name}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f"replay_ctr_{scenario_name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
