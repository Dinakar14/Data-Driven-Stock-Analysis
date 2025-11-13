import pandas as pd
import matplotlib.pyplot as plt

def plot_top_volatility(file="data/stock_summary.csv"):
    df = pd.read_csv(file)
    top_vol = df.nlargest(10, "Volatility")
    plt.bar(top_vol["Symbol"], top_vol["Volatility"])
    plt.xticks(rotation=45)
    plt.title("Top 10 Most Volatile Stocks")
    plt.ylabel("Volatility (Std Dev)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_top_volatility()
