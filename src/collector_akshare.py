import os
import datetime
import json
import pandas as pd
import akshare as ak
from pathlib import Path
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pyarrow.parquet as pq

# ======================
# Cookie 注入与 Requests 劫持 (由 AI4Stock 移植)
# ======================
try:
    from curl_cffi import requests as curleq
except ImportError:
    print("[*] Installing curl_cffi for browser impersonation...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "curl_cffi"])
    from curl_cffi import requests as curleq

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "*/*",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Referer": "https://quote.eastmoney.com/center/gridlist.html",
    "Connection": "keep-alive"
}

class RequestPatcher:
    def __init__(self, cookies_file="data/cookies.json"):
        self.session = curleq.Session(impersonate="chrome120")
        self.session.headers.update(DEFAULT_HEADERS)
        self.cookies_file = cookies_file

    def load_cookies(self):
        if not os.path.exists(self.cookies_file):
            print(f"[!] Warning: {self.cookies_file} not found. Running without custom cookies.")
            return False
        try:
            with open(self.cookies_file, 'r') as f:
                cookie_list = json.load(f)
            cookies = {c['name']: c['value'] for c in cookie_list}
            self.session.cookies.update(cookies)
            print(f"[*] Successfully loaded {len(cookie_list)} cookies.")
            return True
        except Exception as e:
            print(f"[!] Error loading cookies: {e}")
            return False

    def patch(self):
        def patched_get(url, **kwargs):
            kwargs.pop('session', None)
            kwargs.pop('verify', None)
            kwargs.pop('stream', None)
            if 'timeout' not in kwargs: kwargs['timeout'] = 30
            return self.session.get(url, **kwargs)

        def patched_post(url, **kwargs):
            kwargs.pop('session', None)
            kwargs.pop('verify', None)
            kwargs.pop('stream', None)
            if 'timeout' not in kwargs: kwargs['timeout'] = 30
            return self.session.post(url, **kwargs)

        import requests
        requests.get = patched_get
        requests.post = patched_post
        print("[*] Global Requests Hijacked with curl_cffi (Chrome Impersonation).")

# ======================
# 数据采集逻辑
# ======================

# Configuration matching AI4Stock exactly
RAW_DAILY_DIR = Path("data/raw/daily")
RAW_VAL_DIR = Path("data/raw/valuation")
PROCESSED_DIR = Path("data/processed/combined")
QLIB_CSV_DIR = Path("data/qlib_csv_temp") # Temp dir just for dump_bin
QLIB_DIR = Path("data/qlib_data_cn")

DAILY_COLS = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'amount', 'turnover']
VAL_RENAME_MAP = {
    '数据日期': 'date', '当日收盘价': 'v_close', '总市值': 'total_mv', '流通市值': 'circ_mv',
    '总股本': 'total_share', '流通股本': 'circ_share', 'PE(TTM)': 'pe_ttm', 'PE(静)': 'pe_static',
    '市净率': 'pb', 'PEG值': 'peg', '市现率': 'pcf', '市销率': 'ps'
}

# ALL fields that will be dumped into Qlib (including factor which we set to 1.0)
QLIB_FIELDS = ['open', 'high', 'low', 'close', 'volume', 'amount', 'turnover', 
               'total_mv', 'circ_mv', 'total_share', 'circ_share', 
               'pe_ttm', 'pe_static', 'pb', 'peg', 'pcf', 'ps', 'factor']

for d in [RAW_DAILY_DIR, RAW_VAL_DIR, PROCESSED_DIR, QLIB_CSV_DIR, QLIB_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def fetch_stock_list():
    """Fetch all A-share stocks using patched requests."""
    print("[*] Fetching stock list from AkShare...")
    try:
        df = ak.stock_zh_a_spot_em()
        df = df[df["代码"].str.match(r"^(000|001|002|003|300|301|600|601|603|605|688)")]
        return df["代码"].tolist()
    except Exception as e:
        print(f"[!] Error fetching stock list: {e}")
        return []

def save_optimized_parquet(df, path):
    """Save with zstd compression to match old pipeline."""
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype('int32')
    df.to_parquet(path, index=False, engine='pyarrow', compression='zstd')

def fetch_and_fuse(symbol):
    """Fetch daily and valuation data, fuse them, and save."""
    target_end_date = pd.Timestamp("2025-12-31")
    
    # 1. Fetch Daily (HFQ)
    file_path_d = RAW_DAILY_DIR / f"{symbol}.parquet"
    current_start_d = "19900101"
    existing_df_d = None
    
    if file_path_d.exists():
        try:
            existing_df_d = pd.read_parquet(file_path_d)
            if not existing_df_d.empty:
                last_d = pd.to_datetime(existing_df_d['date']).max()
                if last_d < target_end_date:
                    current_start_d = (last_d + pd.Timedelta(days=1)).strftime("%Y%m%d")
                else:
                    current_start_d = None
        except: pass

    if current_start_d:
        try:
            df_new = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=current_start_d, end_date="20251231", adjust="hfq")
            if df_new is not None and not df_new.empty:
                df_new = df_new.rename(columns={"日期": "date", "开盘": "open", "收盘": "close", "最高": "high", "最低": "low", "成交量": "volume", "成交额": "amount", "换手率": "turnover"})
                df_new["date"] = pd.to_datetime(df_new["date"])
                df_new["symbol"] = symbol
                if existing_df_d is not None:
                    df_d = pd.concat([existing_df_d, df_new], ignore_index=True).drop_duplicates(subset=['date']).sort_values('date')
                else:
                    df_d = df_new.sort_values('date')
                save_optimized_parquet(df_d[DAILY_COLS], file_path_d)
        except Exception:
            return False

    # 2. Fetch Valuation
    file_path_v = RAW_VAL_DIR / f"{symbol}.parquet"
    v_need_update = True
    if file_path_v.exists():
        try:
            meta_v = pq.read_metadata(str(file_path_v))
            rg_v = meta_v.row_group(meta_v.num_row_groups - 1)
            max_date_v = pd.Timestamp(rg_v.column(meta_v.schema.names.index('数据日期')).statistics.max)
            if max_date_v >= target_end_date:
                v_need_update = False
        except: pass

    if v_need_update:
        try:
            df_v = ak.stock_value_em(symbol=symbol)
            if df_v is not None and not df_v.empty:
                save_optimized_parquet(df_v, file_path_v)
        except Exception:
            return False

    # 3. Fuse
    try:
        df_d = pd.read_parquet(file_path_d)
        df_v = pd.read_parquet(file_path_v).rename(columns=VAL_RENAME_MAP)
        df_d['date'] = pd.to_datetime(df_d['date'])
        df_v['date'] = pd.to_datetime(df_v['date'])
        
        df = pd.merge(df_d, df_v, on='date', how='outer').sort_values('date')
        if 'v_close' in df.columns:
            df['close'] = df['close'].fillna(df['v_close'])
            df = df.drop(columns=['v_close'])
            
        for c in ['open', 'high', 'low']: 
            if c in df.columns: df[c] = df[c].fillna(df['close'])
        for c in ['volume', 'amount', 'turnover']:
            if c in df.columns: df[c] = df[c].fillna(0.0)
            
        df['symbol'] = symbol
        processed_path = PROCESSED_DIR / f"{symbol}.parquet"
        save_optimized_parquet(df, processed_path)
        return True
    except Exception:
        return False

def convert_to_qlib():
    """Convert combined parquets to CSV temporarily, then run dump_bin.py."""
    print("[*] Generating temporary CSVs with all features for Qlib...")
    files = list(PROCESSED_DIR.glob("*.parquet"))
    
    for f in tqdm(files, desc="Exporting CSV"):
        try:
            df = pd.read_parquet(f)
            # Ensure factor exists
            df['factor'] = 1.0
            
            # Keep date and all required QLIB_FIELDS. Fill missing cols with NaN.
            out_cols = ['date'] + QLIB_FIELDS
            for col in out_cols:
                if col not in df.columns:
                    df[col] = float('nan')
                    
            df[out_cols].to_csv(QLIB_CSV_DIR / f"{f.stem}.csv", index=False)
        except Exception as e:
            pass
            
    script_path = Path("dump_bin.py")
    if not script_path.exists():
        url = "https://raw.githubusercontent.com/microsoft/qlib/main/scripts/dump_bin.py"
        urllib.request.urlretrieve(url, script_path)
    
    cmd = [
        sys.executable, str(script_path), "dump_all",
        "--data_path", str(QLIB_CSV_DIR.resolve()),
        "--qlib_dir", str(QLIB_DIR.resolve()),
        "--include_fields", ",".join(QLIB_FIELDS),
        "--date_field_name", "date"
    ]
    print(f"[*] Running Qlib conversion: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    # Clean up temp CSVs
    import shutil
    shutil.rmtree(QLIB_CSV_DIR)

def collect_data(symbols=None, max_workers=4):
    if symbols is None:
        symbols = fetch_stock_list()

    print(f"[*] Processing {len(symbols)} symbols with {max_workers} workers...")
    success_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_and_fuse, s): s for s in symbols}
        pbar = tqdm(as_completed(futures), total=len(symbols))
        for future in pbar:
            if future.result(): success_count += 1
            pbar.set_postfix({"Success": success_count})
            
    print(f"[+] Data collection finished. {success_count} symbols processed.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Fetch/update all A-share stocks")
    parser.add_argument("--symbols", help="Comma separated symbols to fetch")
    parser.add_argument("--update", action="store_true", help="Fetch missing data for existing symbols in cache via network")
    parser.add_argument("--convert", action="store_true", help="Convert processed Parquets to Qlib binary format")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    
    if args.all or args.symbols or args.update:
        patcher = RequestPatcher()
        patcher.load_cookies()
        patcher.patch()
        
        symbols_to_fetch = []
        if args.symbols:
            symbols_to_fetch = args.symbols.split(",")
        elif args.all:
            symbols_to_fetch = fetch_stock_list()
        elif args.update:
            if PROCESSED_DIR.exists():
                symbols_to_fetch = [f.stem for f in PROCESSED_DIR.glob("*.parquet")]
                
        if symbols_to_fetch:
            collect_data(symbols=symbols_to_fetch, max_workers=args.workers)
        else:
            print("[!] No symbols to update.")
            
    if args.convert:
        convert_to_qlib()
