import requests
import pandas as pd
from datetime import datetime
import os
import csv
import time

# === НАСТРОЙКИ ===
CHECK_INTERVAL_SEC = 60  
SAVE_INTERVAL_SEC = 120  
CATEGORY = "linear"
BASE_URL = "https://api.bybit.com/v5"

INSTRUMENTS = [
    ("BTCUSDT", "Bitcoin"), ("ETHUSDT", "Ethereum"), ("BNBUSDT", "Binance Coin"),
    ("SOLUSDT", "Solana"), ("XRPUSDT", "Ripple"), ("DOGEUSDT", "Dogecoin"),
    ("ADAUSDT", "Cardano"), ("AVAXUSDT", "Avalanche"), ("TRXUSDT", "Tron"),
    ("LINKUSDT", "Chainlink"), ("DOTUSDT", "Polkadot"), ("MATICUSDT", "Polygon"),
    ("LTCUSDT", "Litecoin"), ("SHIBUSDT", "Shiba Inu"), ("UNIUSDT", "Uniswap"),
    ("ATOMUSDT", "Cosmos"), ("ETCUSDT", "Ethereum Classic"), ("NEARUSDT", "Near"),
    ("FILUSDT", "Filecoin"), ("APTUSDT", "Aptos"), ("ARBUSDT", "Arbitrum"),
    ("OPUSDT", "Optimism"), ("INJUSDT", "Injective"), ("SUIUSDT", "Sui"),
    ("PEPEUSDT", "Pepe")
]

# === ФАЙЛЫ ==
PENDING_FILE = "data_pending.csv"
FINAL_FILE = "data_results.csv"


def ensure_csv_headers():
    # ОБНОВЛЕННЫЕ ЗАГОЛОВКИ ДЛЯ PENDING
    if not os.path.exists(PENDING_FILE):
        with open(PENDING_FILE, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "start_time", 
                "ticker",
                "rsi", "ema", "atr",
                "bid_price_pct", "bid_vol_pct",
                "bid_price_5_pct", "bid_vol_5_pct",
                "bid_price_total_pct", "bid_vol_total_pct",
                "min_candles",      # Только 30 минутных
                "hour_candles",     # Отдельно 5 часовых
                "target_up", 
                "target_down", 
                "status"
            ])
            
    # ОБНОВЛЕННЫЕ ЗАГОЛОВКИ ДЛЯ FINAL (без start/end time)
    if not os.path.exists(FINAL_FILE):
        with open(FINAL_FILE, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "ticker",
                "rsi", "ema", "atr",
                "bid_price_pct", "bid_vol_pct",
                "bid_price_5_pct", "bid_vol_5_pct",
                "bid_price_total_pct", "bid_vol_total_pct",
                "min_candles",
                "hour_candles",
                "duration_min", 
                "result"
            ])

def log(text):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {text}")

def candles_list_to_str(df, count):
    """Превращает последние N свечей в строку через запятую"""
    if df is None or df.empty:
        return ""
    closes = df['close'].tail(count).tolist()
    return ",".join(f"{p:.4f}" for p in closes)

def calculate_microstructure(bids_df, asks_df, depth_l5=5):
    try:
        if bids_df.empty or asks_df.empty:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        bids_df = bids_df.sort_values('price', ascending=False).reset_index(drop=True)
        asks_df = asks_df.sort_values('price', ascending=True).reset_index(drop=True)

        b1_p, b1_v = bids_df.iloc[0]['price'], bids_df.iloc[0]['size']
        a1_p, a1_v = asks_df.iloc[0]['price'], asks_df.iloc[0]['size']
        
        l1_price_pct = (b1_p / (b1_p + a1_p)) if (b1_p + a1_p) > 0 else 0.0
        l1_vol_pct = (b1_v / (b1_v + a1_v)) if (b1_v + a1_v) > 0 else 0.0

        top_bids = bids_df.head(depth_l5)
        top_asks = asks_df.head(depth_l5)
        
        val_b_5 = (top_bids['price'] * top_bids['size']).sum()
        val_a_5 = (top_asks['price'] * top_asks['size']).sum()
        vol_b_5 = top_bids['size'].sum()
        vol_a_5 = top_asks['size'].sum()
        
        sum_val_5 = val_b_5 + val_a_5
        sum_vol_5 = vol_b_5 + vol_a_5
        
        l5_price_pct = (val_b_5 / sum_val_5) if sum_val_5 > 0 else 0.0
        l5_vol_pct = (vol_b_5 / sum_vol_5) if sum_vol_5 > 0 else 0.0

        total_b_val = (bids_df['price'] * bids_df['size']).sum()
        total_a_val = (asks_df['price'] * asks_df['size']).sum()
        total_b_vol = bids_df['size'].sum()
        total_a_vol = asks_df['size'].sum()
        
        sum_val_all = total_b_val + total_a_val
        sum_vol_all = total_b_vol + total_a_vol
        
        all_price_pct = (total_b_val / sum_val_all) if sum_val_all > 0 else 0.0
        all_vol_pct = (total_b_vol / sum_vol_all) if sum_vol_all > 0 else 0.0

        return (round(l1_price_pct, 4), round(l1_vol_pct, 4),
                round(l5_price_pct, 4), round(l5_vol_pct, 4),
                round(all_price_pct, 4), round(all_vol_pct, 4))
    except Exception as e:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

def save_new_data(ticker, rsi, ema, atr, metrics, min_candles_str, hour_candles_str, current_price, atr_val):
    """Сохраняет новые данные с разделенными свечами"""
    l1_p, l1_v, l5_p, l5_v, all_p, all_v = metrics
    
    target_up = current_price + (4 * atr_val)
    target_down = current_price - (4 * atr_val)
    
    start_time_str = datetime.now().isoformat()

    with open(PENDING_FILE, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            start_time_str,
            ticker,
            round(rsi, 2), round(ema, 8), round(atr, 8),
            l1_p, l1_v, l5_p, l5_v, all_p, all_v,
            min_candles_str,   # 30 мин
            hour_candles_str,  # 5 час
            round(target_up, 4),
            round(target_down, 4),
            "OPEN"
        ])

def check_and_finalize_positions(current_prices_map):
    if not os.path.exists(PENDING_FILE):
        return 0

    try:
        with open(PENDING_FILE, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception:
        return 0

    remaining_rows = []
    closed_count = 0
    
    fieldnames_pending = [
        "start_time", "ticker", "rsi", "ema", "atr",
        "bid_price_pct", "bid_vol_pct", "bid_price_5_pct", "bid_vol_5_pct",
        "bid_price_total_pct", "bid_vol_total_pct",
        "min_candles", "hour_candles", "target_up", "target_down", "status"
    ]

    for row in rows:
        if row["status"] != "OPEN":
            remaining_rows.append(row)
            continue
            
        ticker = row["ticker"]
        if ticker not in current_prices_map:
            remaining_rows.append(row)
            continue

        current_price = float(current_prices_map[ticker])
        target_up = float(row["target_up"])
        target_down = float(row["target_down"])
        start_time_str = row["start_time"]
        
        result = -1 
        end_time_str = datetime.now().isoformat()
        
        if current_price >= target_up:
            result = 1 
        elif current_price <= target_down:
            result = 0 
        
        if result != -1:
            try:
                start_dt = datetime.fromisoformat(start_time_str)
                end_dt = datetime.fromisoformat(end_time_str)
                duration_sec = (end_dt - start_dt).total_seconds()
                duration_min = round(duration_sec / 60.0, 2)
            except:
                duration_min = 0.0

            # Запись в FINAL_FILE БЕЗ start_time и end_time
            with open(FINAL_FILE, "a", encoding="utf-8", newline="") as f_out:
                writer = csv.writer(f_out)
                writer.writerow([
                    ticker,
                    row["rsi"], row["ema"], row["atr"],
                    row["bid_price_pct"], row["bid_vol_pct"],
                    row["bid_price_5_pct"], row["bid_vol_5_pct"],
                    row["bid_price_total_pct"], row["bid_vol_total_pct"],
                    row["min_candles"],
                    row["hour_candles"],
                    duration_min,
                    result
                ])
            
            log(f"✅ CLOSED {ticker}: Result={result}, Duration={duration_min} min")
            closed_count += 1
        else:
            remaining_rows.append(row)

    if closed_count > 0:
        with open(PENDING_FILE, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames_pending)
            writer.writeheader()
            writer.writerows(remaining_rows)
    
    return closed_count

def calculate_indicators(df, ema_period=20):
    if len(df) < 20: return None, None, None
    close = df['close']
    high = df['high']
    low = df['low']

    ema = close.ewm(span=ema_period, adjust=False).mean().iloc[-1]
    
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi_val = rsi.iloc[-1]

    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14).mean().iloc[-1]

    return ema, rsi_val, atr

def fetch_bybit_data_sync(symbol):
    try:
        # 1. Минутные свечи
        resp_m = requests.get(f"{BASE_URL}/market/kline", 
                              params={"category": CATEGORY, "symbol": symbol, "interval": "1", "limit": "50"},
                              timeout=10)
        data_m = resp_m.json()
        
        if data_m.get('retCode') != 0 or not data_m.get('result', {}).get('list'):
            return None, None, None
            
        kline_list_m = data_m['result']['list']
        df_min = pd.DataFrame(kline_list_m, columns=['time', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
        df_min['time'] = df_min['time'].astype(int)
        df_min[['open', 'high', 'low', 'close', 'volume']] = df_min[['open', 'high', 'low', 'close', 'volume']].astype(float)
        df_min['time'] = pd.to_datetime(df_min['time'], unit='ms')
        df_min.set_index('time', inplace=True)
        df_min = df_min.sort_index()

        # 2. Часовые свечи
        resp_h = requests.get(f"{BASE_URL}/market/kline", 
                              params={"category": CATEGORY, "symbol": symbol, "interval": "60", "limit": "5"},
                              timeout=10)
        data_h = resp_h.json()
        
        df_hour = None
        if data_h.get('retCode') == 0 and data_h.get('result', {}).get('list'):
            kline_list_h = data_h['result']['list']
            df_hour = pd.DataFrame(kline_list_h, columns=['time', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            df_hour['time'] = df_hour['time'].astype(int)
            df_hour[['open', 'high', 'low', 'close', 'volume']] = df_hour[['open', 'high', 'low', 'close', 'volume']].astype(float)
            df_hour['time'] = pd.to_datetime(df_hour['time'], unit='ms')
            df_hour.set_index('time', inplace=True)
            df_hour = df_hour.sort_index()

        # 3. Стакан
        resp_o = requests.get(f"{BASE_URL}/market/orderbook", 
                              params={"category": CATEGORY, "symbol": symbol, "limit": "50"},
                              timeout=10)
        data_o = resp_o.json()
        
        if data_o.get('retCode') != 0:
            return df_min, df_hour, None
            
        ob_result = data_o['result']
        bids_raw = ob_result.get('b', [])
        asks_raw = ob_result.get('a', [])
        
        if not bids_raw or not asks_raw:
            return df_min, df_hour, None
            
        bids_df = pd.DataFrame(bids_raw, columns=['price', 'size']).astype(float)
        asks_df = pd.DataFrame(asks_raw, columns=['price', 'size']).astype(float)
        
        metrics = calculate_microstructure(bids_df, asks_df, depth_l5=5)
        return df_min, df_hour, metrics

    except Exception as e:
        log(f"⚠️ Ошибка загрузки {symbol}: {e}")
        return None, None, None

def run_bot():
    ensure_csv_headers()
    log(f"🚀 ЗАПУСК СБОРА ДАННЫХ. Разделенные свечи (30м | 5ч). Интервал: {SAVE_INTERVAL_SEC}с")
    
    last_save_time = 0
    iteration = 0
    
    while True:
        iteration += 1
        current_time = time.time()
        
        current_prices = {}
        tasks_data = []
        
        for symbol, name in INSTRUMENTS:
            df_min, df_hour, metrics = fetch_bybit_data_sync(symbol)
            
            if df_min is not None and metrics is not None:
                cp = df_min['close'].iloc[-1]
                current_prices[symbol] = cp
                
                ema, rsi, atr = calculate_indicators(df_min)
                
                if ema is not None and atr is not None and atr > 0:
                    # Разделяем строки свечей
                    min_str = candles_list_to_str(df_min, 30)
                    hour_str = candles_list_to_str(df_hour, 5) if df_hour is not None else ""
                    
                    tasks_data.append({
                        "symbol": symbol,
                        "rsi": rsi,
                        "ema": ema,
                        "atr": atr,
                        "metrics": metrics,
                        "min_candles": min_str,
                        "hour_candles": hour_str,
                        "price": cp,
                        "atr_val": atr
                    })
        
        closed_count = check_and_finalize_positions(current_prices)
        if closed_count > 0:
            log(f"Завершено позиций: {closed_count}")
        
        if current_time - last_save_time >= SAVE_INTERVAL_SEC:
            log("⏰ Сохранение новых данных...")
            count_saved = 0
            for item in tasks_data:
                save_new_data(
                    item["symbol"], item["rsi"], item["ema"], item["atr"],
                    item["metrics"], item["min_candles"], item["hour_candles"], 
                    item["price"], item["atr_val"]
                )
                count_saved += 1
            log(f"Сохранено записей: {count_saved}")
            last_save_time = current_time

        sleep_time = max(5, CHECK_INTERVAL_SEC - (time.time() - current_time))
        time.sleep(sleep_time)

if __name__ == "__main__":
    try:
        run_bot()
    except KeyboardInterrupt:
        log("🛑 Остановка пользователем.")