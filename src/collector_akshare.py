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

# Configuration
CACHE_DIR = Path("data/akshare_cache")
CSV_DIR = Path("data/akshare_csv")
QLIB_DIR = Path("data/qlib_data_cn")

for d in [CACHE_DIR, CSV_DIR, QLIB_DIR]:
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

def fetch_and_save_symbol(symbol, start_date="20080101"):
    """Fetch daily data (HFQ - 后复权) and save to CSV."""
    cache_file = CACHE_DIR / f"{symbol}.parquet"
    existing_df = None
    
    if cache_file.exists():
        try:
            existing_df = pd.read_parquet(cache_file)
            if not existing_df.empty:
                last_date = pd.to_datetime(existing_df['date']).max()
                if last_date >= pd.Timestamp.now().normalize() - pd.Timedelta(days=1):
                    # Data is up to date, just sync CSV
                    _sync_to_csv(symbol, existing_df)
                    return True
                start_date = (last_date + pd.Timedelta(days=1)).strftime("%Y%m%d")
        except: pass

    try:
        # 使用 HFQ (后复权)，并将结束日期限制在 2025 年底
        df_new = ak.stock_zh_a_hist(
            symbol=symbol, 
            period="daily", 
            start_date=start_date, 
            end_date="20251231", 
            adjust="hfq"
        )
        
        if df_new is not None and not df_new.empty:
            df_new = df_new.rename(columns={
                "日期": "date", "开盘": "open", "收盘": "close", 
                "最高": "high", "最低": "low", "成交量": "volume", 
                "成交额": "amount", "换手率": "turnover"
            })
            df_new["date"] = pd.to_datetime(df_new["date"])
            
            if existing_df is not None:
                df = pd.concat([existing_df, df_new], ignore_index=True).drop_duplicates(subset=['date']).sort_values('date')
            else:
                df = df_new.sort_values('date')
            
            df.to_parquet(cache_file, index=False)
            _sync_to_csv(symbol, df)
            return True
        return False
    except Exception:
        return False

def _sync_to_csv(symbol, df):
    """Format for Qlib (HFQ prices means factor=1.0)."""
    qlib_df = df[['date', 'open', 'high', 'low', 'close', 'volume']].copy()
    qlib_df['factor'] = 1.0
    qlib_df.to_csv(CSV_DIR / f"{symbol}.csv", index=False)

def import_from_old_project(old_processed_dir):
    """Import existing HFQ data from AI4Stock project."""
    old_dir = Path(old_processed_dir)
    if not old_dir.exists():
        print(f"[!] Old directory {old_dir} does not exist.")
        return

    print(f"[*] Importing from {old_dir}...")
    files = list(old_dir.glob("*.parquet"))
    for f in tqdm(files, desc="Importing"):
        target = CACHE_DIR / f.name
        if not target.exists():
            import shutil
            shutil.copy(f, target)
            try:
                df = pd.read_parquet(target)
                _sync_to_csv(f.stem, df)
            except Exception: pass

def convert_to_qlib():
    """Run dump_bin.py to convert CSVs to Qlib format."""
    script_path = Path("dump_bin.py")
    if not script_path.exists():
        url = "https://raw.githubusercontent.com/microsoft/qlib/main/scripts/dump_bin.py"
        urllib.request.urlretrieve(url, script_path)
    
    cmd = [
        sys.executable, str(script_path), "dump_all",
        "--data_path", str(CSV_DIR.resolve()),
        "--qlib_dir", str(QLIB_DIR.resolve()),
        "--include_fields", "open,high,low,close,volume,factor",
        "--date_field_name", "date"
    ]
    print(f"[*] Running Qlib conversion: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def collect_data(symbols=None, max_workers=4):
    if symbols is None:
        symbols = fetch_stock_list()

    print(f"[*] Processing {len(symbols)} symbols with {max_workers} workers...")
    success_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_and_save_symbol, s): s for s in symbols}
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
    parser.add_argument("--import-old", help="Path to AI4Stock combined data dir to import from")
    parser.add_argument("--update", action="store_true", help="Fetch missing data for existing symbols in cache via network")
    parser.add_argument("--convert", action="store_true", help="Convert CSVs to Qlib binary format")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    
    if args.import_old:
        import_from_old_project(args.import_old)

    if args.all or args.symbols or args.update:
        # 启动劫持
        patcher = RequestPatcher()
        patcher.load_cookies()
        patcher.patch()
        
        symbols_to_fetch = []
        if args.symbols:
            symbols_to_fetch = args.symbols.split(",")
        elif args.all:
            symbols_to_fetch = fetch_stock_list()
        elif args.update:
            # 自动寻找本地已有的缓存文件进行增量更新
            if CACHE_DIR.exists():
                symbols_to_fetch = [f.stem for f in CACHE_DIR.glob("*.parquet")]
                
        if symbols_to_fetch:
            collect_data(symbols=symbols_to_fetch, max_workers=args.workers)
        else:
            print("[!] No symbols to update. Please specify --all, --symbols, or ensure cache directory has files.")
            
    if args.convert:
        convert_to_qlib()
