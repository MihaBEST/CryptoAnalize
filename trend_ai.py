# %%
%matplotlib inline
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ================= 1. ВАША ФУНКЦИЯ НОРМАЛИЗАЦИИ =================
def normalize(prices):
    """
    Ваша функция нормализации стакана/свечей.
    Возвращает список изменений цены относительно спреда.
    """
    if not prices or len(prices) < 2:
        return []
    spread = max(prices) - min(prices)
    if spread == 0:
        return [0.0] * (len(prices) - 1)
    p = spread / 100
    return [round((prices[i] - prices[i-1]) / p, 2) for i in range(1, len(prices))]

# ================= 2. ПОДГОТОВКА ДАННЫХ =================

def calculate_ema_series(prices, span):
    """
    Быстрый расчет EMA для списка цен без pandas (для скорости при обработке строк)
    """
    if len(prices) < span:
        return None

    # Простое среднее для начала
    k = 2 / (span + 1)
    ema_prev = sum(prices[:span]) / span

    # Расчет остальной части
    for price in prices[span:]:
        ema_curr = price * k + ema_prev * (1 - k)
        ema_prev = ema_curr

    return ema_prev

def get_trend_status(min_candles_str):
    """
    Определяет тренд по минутным свечам.
    Возвращает: 1 (Bull), -1 (Bear), 0 (Flat/Filter Out)
    """
    try:
        if not min_candles_str or str(min_candles_str).strip() == "":
            return 0

        prices = [float(x.strip()) for x in str(min_candles_str).split(',') if x.strip()]

        # Нужно хотя бы 30 свечей для корректного EMA(30)
        if len(prices) < 30:
            return 0

        last_price = prices[-1]

        # Расчет EMA(15) и EMA(30) на полном доступном окне
        ema15 = calculate_ema_series(prices, 15)
        ema30 = calculate_ema_series(prices, 30)

        if ema15 is None or ema30 is None:
            return 0

        # Порог "слишком близко" для отсева слабого тренда (опционально)
        # Например, если разница между EMA меньше 0.05% от цены, считаем флэтом
        diff_pct = abs(ema15 - ema30) / last_price

        if diff_pct < 0.0005: # 0.05%
            return 0 # Флэт (EMA слиплись)

        # Проверка условий тренда
        if last_price > ema15 and ema15 > ema30:
            return 1 # Бычий
        elif last_price < ema15 and ema15 < ema30:
            return -1 # Медвежий
        else:
            return 0 # Переходное состояние или шум -> Фильтруем

    except Exception as e:
        return 0

def prepare_data(csv_path='data_results.csv'):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Файл {csv_path} не найден. Загрузите его в Colab.")

    print(f"📂 Загрузка {csv_path}...")
    df = pd.read_csv(csv_path)
    initial_count = len(df)
    print(f"✅ Исходное количество строк: {initial_count}")

    # --- ШАГ 1: ФИЛЬТРАЦИЯ ФЛЭТА ---
    print("🔍 Анализ тренда и удаление флэта...")

    # Применяем функцию определения тренда к каждой строке
    # Используем min_candles, так как там минутные данные
    df['trend_status'] = df['min_candles'].apply(get_trend_status)

    # Оставляем только Бычий (1) и Медвежий (-1). Флэт (0) удаляем.
    df_trend = df[df['trend_status'] != 0].copy()

    filtered_count = initial_count - len(df_trend)
    print(f"⚠️ Отфильтровано строк (Флэт): {filtered_count} ({filtered_count/initial_count*100:.1f}%)")
    print(f"✅ Осталось строк для обучения (Тренд): {len(df_trend)}")

    if len(df_trend) == 0:
        raise ValueError("После фильтрации не осталось данных. Проверьте входной файл.")

    df = df_trend.drop(columns=['trend_status']) # Удаляем вспомогательную колонку
    df.dropna(inplace=True)

    # Вспомогательная функция для получения последней цены из строки свечей
    def get_last_close(candles_str):
        try:
            if not candles_str or str(candles_str).strip() == "":
                return None
            prices = [float(x.strip()) for x in str(candles_str).split(',') if x.strip()]
            if not prices:
                return None
            return prices[-1] # Последняя цена в списке
        except:
            return None

    # Парсинг свечей и применение normalize
    def parse_and_norm(s, target_len):
        try:
            prices = [float(x.strip()) for x in str(s).split(',') if x.strip()]
            if len(prices) < 2: return [0.0] * target_len
            normed = normalize(prices)
            # Обрезаем или дополняем до нужной длины
            if len(normed) > target_len: return normed[-target_len:]
            return normed + [0.0] * (target_len - len(normed))
        except:
            return [0.0] * target_len

    # Применяем к минутным и часовым свечам
    df['min_c'] = df['min_candles'].apply(lambda x: parse_and_norm(x, 30))
    df['hour_c'] = df['hour_candles'].apply(lambda x: parse_and_norm(x, 5))

    # --- ШАГ 2: Нормализация EMA и ATR относительно последней цены ---

    # 1. Получаем последнюю цену закрытия для каждой строки
    df['last_close_price'] = df['min_candles'].apply(get_last_close)

    # 2. Фильтруем строки, где не удалось получить цену (на всякий случай)
    initial_len = len(df)
    df = df[df['last_close_price'].notna() & (df['last_close_price'] > 0)]
    if len(df) < initial_len:
        print(f"⚠️ Удалено {initial_len - len(df)} строк с некорректными данными свечей.")

    # 3. Применяем формулу: (Value / Last_Close) * 100
    df['ema'] = (df['ema'] / df['last_close_price']) * 100
    df['atr'] = (df['atr'] / df['last_close_price']) * 100

    # 4. Временная колонка больше не нужна
    df.drop(columns=['last_close_price'], inplace=True)

    # -----------------------------------------------------------------------

    # Числовые признаки
    num_cols = ['rsi','ema','atr','bid_price_pct','bid_vol_pct',
                'bid_price_5_pct','bid_vol_5_pct','bid_price_total_pct','bid_vol_total_pct']

    # Проверка наличия колонок
    for col in num_cols:
        if col not in df.columns:
            raise ValueError(f"Колонка {col} отсутствует в CSV. Проверьте заголовки.")

    X_num = df[num_cols].values
    X_min = np.stack(df['min_c'].values)
    X_hour = np.stack(df['hour_c'].values)

    # Кодировка тикеров
    le = LabelEncoder()
    X_tick = le.fit_transform(df['ticker']).astype(np.int64)
    print(f"🏷️ Найдено тикеров: {len(le.classes_)}")

    # Целевые переменные
    y_res = df['result'].values.astype(np.float32)

    # Масштабирование числовых признаков
    sc_num = StandardScaler()
    X_num = sc_num.fit_transform(X_num)

    # Разделение на Train/Val (стратифицированное по result)
    # Важно: stratify может вызвать ошибку, если классов слишком мало после фильтрации.
    # Добавим проверку уникальных значений
    unique_classes, counts = np.unique(y_res, return_counts=True)
    if len(unique_classes) < 2 or min(counts) < 2:
         print("⚠️ Недостаточно данных для стратификации, используем обычный split.")
         idx_tr, idx_val = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42)
    else:
        idx_tr, idx_val = train_test_split(
            np.arange(len(df)), test_size=0.2, random_state=42, stratify=y_res
        )

    data = {
        'train': (X_num[idx_tr], X_tick[idx_tr], X_min[idx_tr], X_hour[idx_tr], y_res[idx_tr]),
        'val':   (X_num[idx_val], X_tick[idx_val], X_min[idx_val], X_hour[idx_val], y_res[idx_val]),
        'meta': {'num_scaler': sc_num, 'ticker_enc': le, 'n_tickers': len(le.classes_)}
    }
    return data

# ================= 3. DATASET С АУГМЕНТАЦИЕЙ =================
class SimpleDataset(Dataset):
    def __init__(self, X_num, X_tick, X_min, X_hour, y_res, augment=False):
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        self.X_tick = torch.tensor(X_tick, dtype=torch.long)
        self.X_min = torch.tensor(X_min, dtype=torch.float32)
        self.X_hour = torch.tensor(X_hour, dtype=torch.float32)
        self.y_res = torch.tensor(y_res, dtype=torch.float32)
        self.augment = augment

    def __len__(self): return len(self.y_res)

    def __getitem__(self, i):
        xn, xm, xh = self.X_num[i], self.X_min[i], self.X_hour[i]

        # Аугментация: добавляем небольшой шум только при обучении
        if self.augment:
            xn = xn + torch.randn_like(xn) * 0.025
            xm = xm + torch.randn_like(xm) * 0.02
            xh = xh + torch.randn_like(xh) * 0.015

        return (xn, self.X_tick[i], xm, xh), self.y_res[i]

# ================= 4. МОДЕЛЬ SimpleRobustNet =================
class SimpleRobustNet(nn.Module):
    def __init__(self, n_tickers=25):
        super().__init__()

        # Эмбеддинг тикера (16 признаков)
        self.ticker_emb = nn.Embedding(n_tickers, 16)

        # Сверточная часть (Feature Extractor)
        # Вход: (Batch, 1, 60) -> 30(min) + 5(hour) + 9(num) + 16(emb)
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32), nn.GELU(), nn.Dropout1d(0.2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.GELU(), nn.MaxPool1d(2), # 60 -> 30

            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.GELU(), nn.MaxPool1d(2), # 30 -> 15

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.GELU(),
            nn.AdaptiveAvgPool1d(1) # 15 -> 1
        )

        # Классификационная голова
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(0.5),

            nn.Linear(64, 32),
            nn.GELU(), nn.Dropout(0.3),

            nn.Linear(32, 1)
        )

    def forward(self, x_num, x_tick, x_min, x_hour):
        emb = self.ticker_emb(x_tick)
        # Конкатенация: min(30) + hour(5) + num(9) + emb(16) = 60
        x = torch.cat([x_min, x_hour, x_num, emb], dim=1).unsqueeze(1) # (B, 1, 60)

        features = self.feature_extractor(x).squeeze(2) # (B, 128)
        logit = self.classifier(features) # (B, 1)
        return logit.squeeze(1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# ================= 5. ОБУЧЕНИЕ С ФИЛЬТРОМ УВЕРЕННОСТИ =================
# ================= 5. ОБУЧЕНИЕ С ФИЛЬТРОМ УВЕРЕННОСТИ (ОБНОВЛЕННОЕ) =================
def train_balanced_model(data, model_path='balanced_trader.pth', meta_path='balanced_meta.pkl',
                         epochs=50, lr=1e-3, bs=256, patience=6,
                         conf_threshold=0.75, target_coverage=0.70): # Повысили целевое покрытие до 70%
    """
    Обучение на чистых трендовых данных.
    Цель: Высокая точность (>85%) и хорошее покрытие (>70%).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleRobustNet(n_tickers=data['meta']['n_tickers']).to(device)

    # Расчет веса классов
    y_train_res = data['train'][4]
    pos_count = np.sum(y_train_res == 1)
    neg_count = np.sum(y_train_res == 0)
    # Если классов сильно несбалансированы, вес поможет. Если примерно равно - вес близок к 1.
    pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32).to(device)

    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=2)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    tr_dl = DataLoader(SimpleDataset(*data['train'], augment=True), batch_size=bs, shuffle=True, pin_memory=True)
    vl_dl = DataLoader(SimpleDataset(*data['val'], augment=False), batch_size=bs, shuffle=False, pin_memory=True)

    best_score = -1.0
    trigger = 0
    history = []

    print(f"🚀 Старт обучения (Трендовый режим) | Цель: Acc>85%, Cov~{target_coverage*100}%")

    for ep in range(epochs):
        model.train(); tr_loss = 0
        for x, y in tr_dl:
            x = [v.to(device) for v in x]; y = y.to(device)
            opt.zero_grad()
            logit = model(*x)
            loss = loss_fn(logit, y)
            loss.backward(); opt.step()
            tr_loss += loss.item()

        model.eval(); vl_loss = 0
        correct_confident = 0
        total_confident = 0
        total_samples = len(data['val'][0])

        with torch.no_grad():
            for x, y in vl_dl:
                x = [v.to(device) for v in x]; y = y.to(device)
                logit = model(*x)
                vl_loss += loss_fn(logit, y).item()

                probs = torch.sigmoid(logit)
                preds_class = (probs > 0.5).float()

                # Фильтр уверенности
                mask_conf = (probs >= conf_threshold) | (probs <= (1 - conf_threshold))

                if mask_conf.any():
                    correct_confident += ((preds_class[mask_conf] == y[mask_conf]).sum().item())
                    total_confident += mask_conf.sum().item()

        vl_loss /= len(vl_dl)

        # 1. Сырые метрики
        val_conf_acc = (correct_confident / total_confident) if total_confident > 0 else 0.0
        coverage = total_confident / total_samples

        # 2. РАСЧЕТ BALANCED SCORE (НОВАЯ ФОРМУЛА)

        # Коэффициент покрытия: растет от 0 до 1.
        # Если coverage < target_coverage, штрафуем сильнее.
        # Если coverage >= target_coverage, считаем равным 1.0 (или чуть больше для бонуса)
        cov_factor = min(coverage / target_coverage, 1.0)

        # Новая формула: Акцент на точности (70% веса) и покрытии (30% веса)
        # Раньше было: Acc * (0.2 + 0.8 * cov). Теперь точность важнее.
        balanced_score = val_conf_acc * (0.4 + 0.6 * cov_factor)

        # Альтернатива для очень строгой точности:
        # balanced_score = (val_conf_acc ** 1.5) * (cov_factor ** 0.5)

        current_lr = opt.param_groups[0]['lr']
        sched.step(balanced_score)

        history.append((ep+1, tr_loss/len(tr_dl), vl_loss, val_conf_acc, coverage, balanced_score, current_lr))

        status = "✅ BEST" if balanced_score > best_score else ""
        if balanced_score > best_score:
            best_score = balanced_score
            torch.save(model.state_dict(), model_path)
            trigger = 0
        else:
            trigger += 1

        print(f"Ep {ep+1:02d} | Loss: {vl_loss:.4f} | Acc: {val_conf_acc:.4f} | Cov: {coverage*100:.1f}% | Score: {balanced_score:.4f} {status}")

        if trigger >= patience:
            print("⏹ Early Stopping (Score не растет)")
            break

    with open(meta_path, 'wb') as f:
        pickle.dump(data['meta'], f)

    plot_balanced_history(history, target_coverage)
    print(f"\n🏆 Лучший Balanced Score: {best_score:.4f}")
    return model, history

def plot_balanced_history(history, target_cov):
    epochs = [h[0] for h in history]
    acc = [h[3] for h in history]
    cov = [h[4] for h in history]
    score = [h[5] for h in history]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # График Скора
    ax1.plot(epochs, score, 'g-o', label='Balanced Score', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Balanced Score', color='g')
    ax1.tick_params(axis='y', labelcolor='g')
    ax1.grid(True, alpha=0.3)

    # Вторичная ось
    ax2 = ax1.twinx()
    ax2.plot(epochs, acc, 'b--', label='Confident Acc')
    ax2.plot(epochs, cov, 'r--', label='Coverage')
    ax2.axhline(y=target_cov, color='gray', linestyle=':', label=f'Target Cov ({target_cov*100}%)')
    ax2.set_ylabel('Accuracy / Coverage', color='b')
    ax2.tick_params(axis='y', labelcolor='b')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right')

    plt.title('Баланс Точности и Покрытия')
    plt.tight_layout()
    plt.show()

def plot_balanced_history(history):
    epochs = [h[0] for h in history]
    acc = [h[3] for h in history]
    cov = [h[4] for h in history]
    score = [h[5] for h in history]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(epochs, score, 'g-o', label='Balanced Score (Acc * sqrt(Cov))', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Balanced Score', color='g')
    ax1.tick_params(axis='y', labelcolor='g')
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(epochs, acc, 'b--', label='Confident Acc')
    ax2.plot(epochs, cov, 'r--', label='Coverage')
    ax2.set_ylabel('Accuracy / Coverage', color='b')
    ax2.tick_params(axis='y', labelcolor='b')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.title('Обучение с балансом Точности и Покрытия')
    plt.tight_layout()
    plt.show()

# ================= 7. ЗАПУСК =================
if __name__ == '__main__':
    # Укажите путь к вашему файлу
    CSV_FILE = '/content/drive/MyDrive/Buy Bit bot/data_results.csv'

    try:
        data = prepare_data(CSV_FILE)
        model, history = train_balanced_model(
            data,
            model_path='/content/drive/MyDrive/Buy Bit bot/trend_robust_netv3.pth',
            meta_path='/content/drive/MyDrive/Buy Bit bot/trend_metav3.pkl',
            epochs=40,
            lr=1e-3,
            bs=256,
            conf_threshold=0.75 # Предсказания считаются только если prob > 0.75 или < 0.25
        )
    except Exception as e:
        print(f"❌ Ошибка: {e}")