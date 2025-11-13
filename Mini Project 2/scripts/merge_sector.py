# scripts/merge_sector.py
import os
import sys
import pandas as pd

def detect_and_normalize_sector_file(path):
    """Read sector CSV and normalize its symbol/ticker column to plain tickers."""
    df = pd.read_csv(path, dtype=str)
    df.columns = [c.strip() for c in df.columns]

    # find possible symbol column (case-insensitive)
    col_map = {c.lower(): c for c in df.columns}
    symbol_col = None
    if "symbol" in col_map:
        symbol_col = col_map["symbol"]
    elif "ticker" in col_map:
        symbol_col = col_map["ticker"]
    else:
        # fallback: choose the last column (your file has COMPANY,sector,Symbol so last is Symbol)
        symbol_col = df.columns[-1]

    # find possible sector column
    sector_col = None
    for cand in ("sector", "Sector", "industry", "Industry"):
        if cand in df.columns:
            sector_col = cand
            break
    if sector_col is None:
        # try lowercase map
        if "sector" in col_map:
            sector_col = col_map["sector"]

    if symbol_col is None or sector_col is None:
        raise ValueError(f"Could not autodetect symbol or sector column. Found columns: {list(df.columns)}")

    # Extract ticker: if symbol column contains "NAME: TICKER", take part after colon
    def extract_ticker(val: str):
        if pd.isna(val):
            return ""
        s = str(val).strip()
        if ":" in s:
            # take right side of last colon
            parts = s.split(":")
            ticker = parts[-1]
        else:
            ticker = s
        # remove extra punctuation and whitespace
        ticker = ticker.strip().upper()
        # remove surrounding quotes if any
        if ticker.startswith('"') and ticker.endswith('"'):
            ticker = ticker[1:-1].strip()
        return ticker

    df["TickerNorm"] = df[symbol_col].apply(extract_ticker)
    df[sector_col] = df[sector_col].astype(str).str.strip()
    # Keep only necessary columns
    out = df[["TickerNorm", sector_col]].rename(columns={"TickerNorm": "Symbol", sector_col: "Sector"})
    return out

def main():
    project_root = os.getcwd()
    print("Project root:", project_root)

    sector_path = os.path.join("data", "sector_data.csv")
    if not os.path.exists(sector_path):
        print(f"ERROR: {sector_path} not found. Please save your CSV as data/sector_data.csv")
        sys.exit(1)

    stock_summary_path = os.path.join("data", "stock_summary.csv")
    if not os.path.exists(stock_summary_path):
        print(f"ERROR: {stock_summary_path} not found. Run scripts/data_analysis.py first.")
        sys.exit(2)

    try:
        sectors = detect_and_normalize_sector_file(sector_path)
    except Exception as e:
        print("Failed to parse sector file:", e)
        sys.exit(3)

    print(f"Loaded {len(sectors)} rows from sector file. Example rows:")
    print(sectors.head(5).to_string(index=False))

    # Load stock summary
    stocks = pd.read_csv(stock_summary_path, dtype=str)
    # ensure Symbol exists
    if "Symbol" not in stocks.columns:
        # attempt to find symbol-like column
        lower_cols = {c.lower(): c for c in stocks.columns}
        for cand in ("symbol", "ticker"):
            if cand in lower_cols:
                stocks = stocks.rename(columns={lower_cols[cand]: "Symbol"})
                break
    if "Symbol" not in stocks.columns:
        print("ERROR: data/stock_summary.csv has no Symbol column. Columns:", list(stocks.columns))
        sys.exit(4)

    # Normalize stocks' Symbol column
    stocks["Symbol"] = stocks["Symbol"].astype(str).str.strip().str.upper()

    # Ensure YearlyReturn is numeric
    if "YearlyReturn" in stocks.columns:
        stocks["YearlyReturn"] = pd.to_numeric(stocks["YearlyReturn"], errors="coerce")
    else:
        print("WARNING: YearlyReturn column not found in stock_summary.csv. Sector averages will be NaN.")

    # Normalize sector symbols
    sectors["Symbol"] = sectors["Symbol"].astype(str).str.strip().str.upper()

    # Merge
    merged = stocks.merge(sectors, on="Symbol", how="left")

    # Count missing sectors
    missing_sector_count = merged["Sector"].isna().sum()
    total = len(merged)
    print(f"Merged. {missing_sector_count}/{total} symbols have no sector match (they will be labeled 'Unknown').")

    merged["Sector"] = merged["Sector"].fillna("Unknown")

    # Compute sector average yearly return (skip NaNs)
    sector_summary = merged.groupby("Sector", as_index=False)["YearlyReturn"].mean()
    # Sort descending (best sectors first)
    sector_summary = sector_summary.sort_values("YearlyReturn", ascending=False)

    out_dir = "data"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "sector_performance.csv")
    sector_summary.to_csv(out_path, index=False)
    print(f"✅ Saved sector performance to {out_path}")

    # Save merged mapping (optional) to inspect which tickers didn't match
    merged_out = os.path.join(out_dir, "stock_summary_with_sector.csv")
    merged.to_csv(merged_out, index=False)
    print(f"Saved merged stock->sector mapping to {merged_out}")
    # If many missing, show examples
    if missing_sector_count > 0:
        missing_examples = merged[merged["Sector"]=="Unknown"].head(10)[["Symbol"]]
        print("Examples of symbols without a sector match (first 10):")
        print(missing_examples.to_string(index=False))

if __name__ == "__main__":
    main()
