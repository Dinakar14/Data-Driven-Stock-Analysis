# scripts/data_analysis.py
import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path

def ensure_csv_folder(csv_folder="data/csv_data"):
    """Ensure csv_folder exists. Return True if contains at least one CSV file."""
    os.makedirs(csv_folder, exist_ok=True)
    files = list(Path(csv_folder).glob("*.csv"))
    return len(files) > 0

def split_combined_file(combined_path, csv_folder="data/csv_data"):
    """
    If a combined CSV exists (data/all_stocks.csv) with a Symbol column,
    split it into per-symbol CSVs under csv_folder and return True.
    """
    if not os.path.exists(combined_path):
        return False

    try:
        df = pd.read_csv(combined_path)
    except Exception as e:
        print(f"Failed to read {combined_path}: {e}")
        return False

    if 'Symbol' not in df.columns:
        print(f"{combined_path} found but has no 'Symbol' column; cannot split.")
        return False

    # Normalize column names
    df.columns = [c.strip() for c in df.columns]

    # For each symbol, write CSV to csv_folder
    for sym, g in df.groupby('Symbol'):
        sym_clean = str(sym).strip().upper()
        out_path = os.path.join(csv_folder, f"{sym_clean}.csv")
        g.to_csv(out_path, index=False)
        print(f"Created per-symbol CSV: {out_path}")

    return True

def create_sample_csvs(csv_folder="data/csv_data"):
    """Create two tiny sample CSVs so pipeline can run end-to-end for testing."""
    os.makedirs(csv_folder, exist_ok=True)
    sample1 = """Date,Open,High,Low,Close,Volume,Symbol
2024-01-01,2500,2550,2480,2520,1500000,RELIANCE
2024-01-02,2520,2560,2510,2550,1400000,RELIANCE
2024-01-03,2550,2600,2540,2580,1600000,RELIANCE
2024-01-04,2580,2620,2570,2600,1550000,RELIANCE
2024-01-05,2600,2650,2590,2630,1700000,RELIANCE
"""
    sample2 = """Date,Open,High,Low,Close,Volume,Symbol
2024-01-01,3200,3260,3180,3220,800000,TCS
2024-01-02,3220,3280,3210,3250,820000,TCS
2024-01-03,3250,3300,3240,3280,780000,TCS
2024-01-04,3280,3330,3270,3310,790000,TCS
2024-01-05,3310,3350,3300,3340,810000,TCS
"""
    p1 = os.path.join(csv_folder, "RELIANCE.csv")
    p2 = os.path.join(csv_folder, "TCS.csv")
    with open(p1, "w", encoding="utf-8") as f:
        f.write(sample1)
    with open(p2, "w", encoding="utf-8") as f:
        f.write(sample2)
    print(f"Sample CSVs created: {p1}, {p2}")
    return True

def load_symbol_csvs(csv_folder):
    """Load per-symbol CSVs from folder and return dict symbol->df (with Date parsed if present)."""
    files = glob.glob(os.path.join(csv_folder, "*.csv"))
    stock_dfs = {}
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"Warning: failed to read {f}: {e}")
            continue

        # Normalize column names (strip whitespace)
        df.columns = [c.strip() for c in df.columns]

        # If Symbol column missing, infer from filename
        if 'Symbol' not in df.columns:
            symbol_from_file = os.path.splitext(os.path.basename(f))[0]
            df['Symbol'] = symbol_from_file

        # Parse Date if exists
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

        # Ensure numeric columns where expected
        for col in ['Open','High','Low','Close','Volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Uppercase and strip symbol
        sym = str(df['Symbol'].iat[0]).strip().upper()
        stock_dfs[sym] = df
    return stock_dfs

def compute_stock_summary(stock_dfs, out_path="data/stock_summary.csv"):
    rows = []
    for sym, df in stock_dfs.items():
        # require Close column to compute metrics
        if 'Close' not in df.columns:
            print(f"Warning: {sym} missing Close column — skipping.")
            continue

        # ensure Close numeric
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')

        # daily return
        df['DailyReturn'] = df['Close'].pct_change()

        # yearly return: use first and last non-null close
        valid_close = df['Close'].dropna()
        if valid_close.shape[0] < 2:
            yearly_return = np.nan
        else:
            first = valid_close.iloc[0]
            last = valid_close.iloc[-1]
            yearly_return = (last - first) / first if first != 0 else np.nan

        volatility = df['DailyReturn'].std(ddof=0)
        avg_price = df['Close'].mean()
        avg_volume = pd.to_numeric(df['Volume'], errors='coerce').mean() if 'Volume' in df.columns else np.nan

        rows.append({
            "Symbol": sym,
            "YearlyReturn": yearly_return,
            "Volatility": volatility,
            "AvgPrice": avg_price,
            "AvgVolume": avg_volume
        })

    summary = pd.DataFrame(rows, columns=["Symbol","YearlyReturn","Volatility","AvgPrice","AvgVolume"])
    # coerce numeric types
    for col in ["YearlyReturn","Volatility","AvgPrice","AvgVolume"]:
        if col in summary.columns:
            summary[col] = pd.to_numeric(summary[col], errors='coerce')

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    summary.to_csv(out_path, index=False)
    print(f"Saved {out_path}")
    return summary

def build_close_matrix(stock_dfs, out_path="data/close_matrix.csv"):
    frames = []
    for sym, df in stock_dfs.items():
        if 'Date' not in df.columns or 'Close' not in df.columns:
            continue
        tmp = df[['Date','Close']].copy()
        tmp = tmp.set_index('Date').rename(columns={'Close': sym})
        frames.append(tmp)

    if not frames:
        combined = pd.DataFrame()
        combined.to_csv(out_path)
        print(f"Saved {out_path} (empty)")
        return combined

    combined = pd.concat(frames, axis=1, sort=True)
    combined = combined.sort_index()
    # Forward/backfill to reduce NaNs where possible
    combined = combined.fillna(method='ffill').fillna(method='bfill')
    combined.to_csv(out_path)
    print(f"Saved {out_path}")
    return combined

def compute_cumulative_returns(close_matrix, out_path="data/cumulative_returns.csv"):
    if close_matrix.empty:
        cum = pd.DataFrame()
        cum.to_csv(out_path)
        print(f"Saved {out_path} (empty)")
        return cum
    # Use first available row as base
    first_row = close_matrix.iloc[0].replace(0, np.nan)
    cum = (close_matrix / first_row) - 1
    cum.to_csv(out_path)
    print(f"Saved {out_path}")
    return cum

def compute_correlation_matrix(close_matrix, out_path="data/correlation_matrix.csv"):
    if close_matrix.empty:
        corr = pd.DataFrame()
        corr.to_csv(out_path)
        print(f"Saved {out_path} (empty)")
        return corr
    pct = close_matrix.pct_change().dropna(how='all')
    corr = pct.corr()
    corr.to_csv(out_path)
    print(f"Saved {out_path}")
    return corr

def compute_sector_performance(summary_df, sector_csv_path="data/sector_data.csv", out_path="data/sector_performance.csv"):
    # If sector file exists, load and normalize tickers; else mark all Unknown and compute single mean
    if os.path.exists(sector_csv_path):
        sectors = pd.read_csv(sector_csv_path, dtype=str)
        sectors.columns = [c.strip() for c in sectors.columns]

        # Detect symbol and sector columns
        col_map = {c.lower(): c for c in sectors.columns}
        symbol_col = col_map.get('symbol') or col_map.get('ticker') or sectors.columns[-1]
        sector_col = col_map.get('sector') or col_map.get('industry') or sectors.columns[1] if len(sectors.columns) > 1 else None

        def extract_ticker(s):
            if pd.isna(s):
                return ""
            s = str(s).strip()
            if ":" in s:
                s = s.split(":")[-1]
            s = s.split("(")[0]
            s = s.replace('"','').strip().upper()
            s = s.strip()
            return s

        sectors['SymbolNorm'] = sectors[symbol_col].apply(extract_ticker)
        sectors['SectorNorm'] = sectors[sector_col].astype(str).str.strip() if sector_col else "Unknown"
        sectors_norm = sectors[['SymbolNorm','SectorNorm']].rename(columns={'SymbolNorm':'Symbol','SectorNorm':'Sector'})

        merged = summary_df.copy()
        merged['SymbolNorm'] = merged['Symbol'].astype(str).str.strip().str.upper()
        merged = merged.merge(sectors_norm, left_on='SymbolNorm', right_on='Symbol', how='left')
        merged['Sector'] = merged['Sector'].fillna('Unknown')
        merged['YearlyReturn'] = pd.to_numeric(merged['YearlyReturn'], errors='coerce')
        sector_summary = merged.groupby('Sector', as_index=False)['YearlyReturn'].mean().sort_values('YearlyReturn', ascending=False)
        sector_summary.to_csv(out_path, index=False)
        print(f"Saved {out_path}")
        return sector_summary
    else:
        tmp = summary_df.copy()
        tmp['YearlyReturn'] = pd.to_numeric(tmp['YearlyReturn'], errors='coerce')
        mean_return = tmp['YearlyReturn'].mean()
        sector_summary = pd.DataFrame([{'Sector':'Unknown','YearlyReturn':mean_return}])
        sector_summary.to_csv(out_path, index=False)
        print(f"Saved {out_path} (all Unknown)")
        return sector_summary

def main():
    csv_folder = "data/csv_data"
    combined_path = "data/all_stocks.csv"

    # 1) if csv_folder has files -> proceed
    has_files = ensure_csv_folder(csv_folder)
    if not has_files:
        # try to split combined file
        if os.path.exists(combined_path):
            print(f"No per-symbol CSVs found. Splitting combined file: {combined_path}")
            ok = split_combined_file(combined_path, csv_folder=csv_folder)
            if not ok:
                print("Failed to split combined file. Creating sample CSVs for testing.")
                create_sample_csvs(csv_folder=csv_folder)
        else:
            # create samples automatically so pipeline can proceed
            print("No per-symbol CSVs and no data/all_stocks.csv found. Creating sample CSVs for testing.")
            create_sample_csvs(csv_folder=csv_folder)

    # Now load symbol CSVs
    stock_dfs = load_symbol_csvs(csv_folder)
    if not stock_dfs:
        print(f"No valid symbol CSVs found in {csv_folder}.")
        return

    # 1) stock summary
    summary = compute_stock_summary(stock_dfs, out_path="data/stock_summary.csv")

    # 2) close matrix
    close_matrix = build_close_matrix(stock_dfs, out_path="data/close_matrix.csv")

    # 3) cumulative returns
    cum = compute_cumulative_returns(close_matrix, out_path="data/cumulative_returns.csv")

    # 4) correlation matrix
    corr = compute_correlation_matrix(close_matrix, out_path="data/correlation_matrix.csv")

    # 5) sector performance
    sector = compute_sector_performance(summary, sector_csv_path="data/sector_data.csv", out_path="data/sector_performance.csv")

if __name__ == "__main__":
    main()
