# Context Bandits: LinUCB & Linear Thompson Sampling

Исследование эффективности линейных контекстных алгоритмов многоруких бандитов (LinUCB и Linear Thompson Sampling) в задаче персонализированной рекомендации фильмов. Проект разработан в рамках учебной практики.

## Особенности проекта
- **Реализация алгоритмов**: LinUCB и Linear Thompson Sampling (disjoint-архитектура) с теоретическими гарантиями сублинейного роста сожаления $O(d\sqrt{T \ln T})$.
- **Синтетическая среда**: Настраиваемый генератор для валидации в сценариях `strong`, `weak` и `noise` зависимости от контекста.
- **Оценка на реальных данных**: Метод непредвзятой офлайн-оценки (unbiased offline evaluation / replay with recycling) на датасете **MovieLens 1M**.
- **Воспроизводимость**: Фиксация `seed`, расчет 95% доверительных интервалов, использование современного менеджера пакетов `uv`.

## Установка и запуск

### Требования
- Python 3.13+
- Менеджер пакетов `uv` (рекомендуется) или стандартный `pip`

### Установка зависимостей
```bash
# Клонирование репозитория
git clone https://github.com/PavlovIO/Context_Bandits.git
cd Context_Bandits

# Установка зависимостей через uv (рекомендуется, быстро и надежно)
uv sync

# Альтернатива через стандартный pip
pip install -r requirements.txt
```
### Запуск экспериментов
Основной скрипт main.py последовательно запускает:
Синтетические эксперименты (3 сценария, 20 независимых запусков).
Replay-эксперимент на датасете MovieLens 1M (топ-30 фильмов, 28 демографических признаков).
```bash
python main.py
```
Результаты (графики .png и сохраненные метрики .pkl) автоматически сохраняются в директорию results/.

## Структура проекта
.
├── bandits/               # Реализация политик (LinUCB, LinTS, UCB1, TS)
├── bandit_environment/    # Модели сред (SyntheticEnvironment)
├── data_loading/          # Скрипты загрузки и парсинга MovieLens
├── features/              # Feature engineering
├── experiments/           # Логика проведения, построение графиков и сохранения экспериментов
├── results/               # Сохраненные метрики (.pkl) и итоговые графики (.png)
├── MovieLensData/         # Исходные данные
├── main.py                # Точка входа для запуска всех экспериментов
├── DataAnalysis.ipynb     # EDA и визуальный анализ датасета
├── pyproject.toml         # Конфигурация проекта и зависимости
└── README.md              # Этот файл

## Ключевые результаты
На датасете MovieLens 1M при горизонте планирования 3000 шагов:
- **Baseline** (случайная политика): CTR ≈ 0.454
- **LinUCB** (alpha=0.5): CTR ≈ 0.609 (+34.2% к baseline)
- **Linear TS** (v=0.15): CTR ≈ 0.622 (+37.0% к baseline)
Контекстные алгоритмы превосходят неконтекстные аналоги (UCB1, TS) на 9–16% по итоговому CTR, что подтверждает их эффективность в условиях холодного старта.
## Основные источники
1. Li et al., A Contextual-Bandit Approach to Personalized News Article Recommendation, WWW 2010.
2. Agrawal & Goyal, Thompson Sampling for Contextual Bandits with Linear Payoffs, ICML 2013.
3. Harper & Konstan, The MovieLens Datasets: History and Context, TIIS 2015.