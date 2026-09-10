import pickle
import os
from typing import Dict, List, Any, Optional

def save_results(results: Dict[str, List[Dict[str, Any]]], 
                 filepath: str,
                 metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Сохраняет результаты экспериментов в pickle файл.
    
    Args:
        results: Словарь {algo_name: [stats_run_1, stats_run_2, ...]}
        filepath: Путь к файлу (например, 'results/experiment_1.pkl')
        metadata: Опциональные метаданные
    """
    # Создаем папку, если её нет
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    data_to_save = {
        'results': results,
        'metadata': metadata or {}
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(data_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Результаты сохранены: {filepath}")


def load_results(filepath: str) -> Dict[str, Any]:
    """
    Загружает результаты экспериментов из pickle файла.

    Returns:
        Словарь с ключами 'results' и 'metadata'
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Файл не найден: {filepath}")
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    return data