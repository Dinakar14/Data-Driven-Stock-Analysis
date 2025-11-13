# scripts/fix_and_generate.py
import os
import glob
import pandas as pd
import numpy as np

CSV_FOLDER = os.path.join("data", "csv_data")
OUT_DIR = "data"

def find_csv_files():
    return sorted(glob.glob(os.path.join(CSV_FOLDER, "*.csv")))

def normalize_columns(df):
    # Normalize column names: strip, title-case common names
    cols = {c: c.strip() for c in df.columns}
    df = df.rename(columns=cols)
    # mapping common lower-case names to TitleCase expected ones
    mapping = {}
    for c in df.columns:
        lc = c.lower()
        if lc in ("date", "day"):
            mapping[c] = "Date"
        elif lc in ("open", "o"):
            mapping[c] = "Open"
        elif lc in ("high", "h"):
            mapping[c] = "High"
        elif lc in ("low", "l"):
            mapping[c] = "Low"
        elif lc in ("close", "c", "adjclose", "adj_close"):
            mapping[c] = "Close"
        elif lc in ("volume", "vol"):
            mapping[c] = "Volume"
        elif lc in ("symbol", "ticker"):
            mapping[c] = "Symbol"
    if mapping:
        df = df.rename(columns=mapping)
    return df

def try_parse_dates(df):
    if 'Date' not in df.columns:
        return df
    # if dates are already datetime, return
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        # try many common formats
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=False)
        # try dayfirst if many NaT
        na_ratio = df['Date'].isna().mean()
        if na_ratio > 0.5:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)
    return df

def inspect_and_fix_all():
    files = find_csv_files()
    if not files:
        print("No per-symbol CSVs found in data/csv_data. Run data_extraction.py first or add CSVs.")
        return None

    fixed_dfs = {}
    issues = {}
    for f in files:
        name = os.path.basename(f)
        try:
            df = pd.read_csv(f)
        except Exception as e:
            issues[name] = f"read_error: {e}"
            print(f"[ERROR] Could not read {name}: {e}")
            continue

        df = normalize_columns(df)
        df = try_parse_dates(df)

        missing = []
        if 'Close' not in df.columns:
            missing.append('Close')
        if 'Date' not in df.columns:
            missing.append('Date')

        # If symbol missing, infer from filename
        if 'Symbol' not in df.columns:
            inferred = os.path.splitext(name)[0].upper()
            df['Symbol'] = inferred

        # Print top 3 rows for quick inspection
        sample = df.head(3).to_dict(orient='records')
        print(f"\nFile: {name}  -> Symbol: {df['Symbol'].iat[0]}")
        print(" Columns:", list(df.columns))
        print(" Missing:", missing if missing else "None")
        print(" Sample rows (up to 3):")
        for r in sample:
            print("  ", r)

        if missing:
            issues[name] = f"missing:{','.join(missing)}"
            # skip adding to fixed_dfs if critical columns missing
            continue

        # ensure numeric for Close and Volume
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        if 'Volume' in df.columns:
            df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')

        # drop rows with no Date or no Close after coercion
        if 'Date' in df.columns:
            df = df.dropna(subset=['Date', 'Close']).copy()
            # sort
            df = df.sort_values('Date').reset_index(drop=True)
        fixed_dfs[df['Symbol'].iat[0]] = df

    print("\nSummary of issues found:")
    if not issues:
        print("  No issues detected.")
    else:
        for k,v in issues.items():
            print(" ", k, "->", v)
    print(f"\nFixed and loadable symbols count: {len(fixed_dfs)}")
    return fixed_dfs

def build_close_matrix(stock_dfs, out_path=os.path.join(OUT_DIR,"close_matrix.csv")):
    frames = []
    for sym, df in stock_dfs.items():
        tmp = df[['Date','Close']].copy()
        tmp = tmp.rename(columns={'Close': sym})
        tmp = tmp.set_index('Date')
        frames.append(tmp)
    if not frames:
        print("No valid close frames to build close_matrix.")
        return None
    combined = pd.concat(frames, axis=1).sort_index()
    # fill forward then back
    combined = combined.fillna(method='ffill').fillna(method='bfill')
    combined.to_csv(out_path)
    print(f"Saved {out_path} (shape={combined.shape})")
    return combined

def compute_cumulative_returns(close_matrix, out_path=os.path.join(OUT_DIR,"cumulative_returns.csv")):
    if close_matrix is None or close_matrix.empty:
        print("Close matrix empty; cannot compute cumulative returns.")
        return None
    cum = (close_matrix / close_matrix.iloc[0]) - 1
    cum.to_csv(out_path)
    print(f"Saved {out_path}")
    return cum

def compute_correlation(close_matrix, out_path=os.path.join(OUT_DIR,"correlation_matrix.csv")):
    if close_matrix is None or close_matrix.empty:
        print("Close matrix empty; cannot compute correlation.")
        return None
    pct = close_matrix.pct_change().dropna(how='all')
    corr = pct.corr()
    corr.to_csv(out_path)
    print(f"Saved {out_path}")
    return corr

def try_sector_performance(summary_path=os.path.join(OUT_DIR,"stock_summary.csv"), sector_csv="sector_data.csv", out_path=os.path.join(OUT_DIR,"sector_performance.csv")):
    if not os.path.exists(sector_csv):
        print(f"{sector_csv} missing. Create a mapping file if you want sector performance.")
        return None
    if not os.path.exists(summary_path):
        print(f"{summary_path} missing. run compute summary first.")
        return None
    summary = pd.read_csv(summary_path)
    # ensure numeric
    summary['YearlyReturn'] = pd.to_numeric(summary.get('YearlyReturn', pd.Series()), errors='coerce')
    sectors = pd.read_csv(sector_csv)
    merged = summary.merge(sectors, on='Symbol', how='left')
    merged['Sector'] = merged['Sector'].fillna('Unknown')
    sector_summary = merged.groupby('Sector')['YearlyReturn'].mean().reset_index().sort_values('YearlyReturn', ascending=False)
    sector_summary.to_csv(out_path, index=False)
    print(f"Saved {out_path}")
    return sector_summary

if __name__ == "__main__":
    fixed = inspect_and_fix_all()
    if not fixed:
        print("No valid symbol data to process. Fix input CSVs and re-run.")
        raise SystemExit(1)
    close = build_close_matrix(fixed)
    compute_cumulative_returns(close)
    compute_correlation(close)
    try_sector_performance()
    print("Done.")
