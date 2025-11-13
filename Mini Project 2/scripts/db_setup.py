# scripts/db_setup.py
import os
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

def upload_to_mysql_or_sqlite(csv_file, mysql_url=None, sqlite_path="data/stock.db"):
    df = pd.read_csv(csv_file)
    # Ensure numeric types preserved
    for col in ['YearlyReturn','Volatility','AvgPrice','AvgVolume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Try MySQL if provided
    if mysql_url:
        try:
            engine = create_engine(mysql_url, pool_pre_ping=True)
            df.to_sql("stock_summary", engine, if_exists="replace", index=False)
            print("Data uploaded to MySQL!")
            return
        except SQLAlchemyError as e:
            print("MySQL upload failed:", str(e))

    # Fallback to SQLite
    os.makedirs(os.path.dirname(sqlite_path) or ".", exist_ok=True)
    sqlite_url = f"sqlite:///{sqlite_path}"
    engine = create_engine(sqlite_url)
    df.to_sql("stock_summary", engine, if_exists="replace", index=False)
    print(f"Data uploaded to SQLite at {sqlite_path}")

if __name__ == "__main__":
    mysql_url = os.environ.get("STOCK_MYSQL_URL", None)
    upload_to_mysql_or_sqlite("data/stock_summary.csv", mysql_url=mysql_url)
