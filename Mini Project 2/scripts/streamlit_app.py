# scripts/streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

st.set_page_config(layout="wide", page_title="Data-Driven Stock Analysis")

st.title("📈 Data-Driven Stock Analysis Dashboard")

summary_path = "data/stock_summary.csv"
cum_path = "data/cumulative_returns.csv"
corr_path = "data/correlation_matrix.csv"
sector_path = "data/sector_performance.csv"
close_matrix_path = "data/close_matrix.csv"

if not os.path.exists(summary_path):
    st.error("Run `python scripts/data_analysis.py` first to generate analysis artifacts.")
    st.stop()

# Load summary
df = pd.read_csv(summary_path)

# Convert numeric columns safely
for col in ["YearlyReturn", "Volatility", "AvgPrice", "AvgVolume"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

st.sidebar.header("Filters")
min_return = st.sidebar.slider("Minimum Yearly Return", -1.0, 3.0, -1.0, step=0.01)
sector_filter = None
if os.path.exists("sector_data.csv"):
    sector_map = pd.read_csv("sector_data.csv")
    sectors = ['All'] + sorted(sector_map['Sector'].dropna().unique().tolist())
    sector_filter = st.sidebar.selectbox("Sector filter", sectors, index=0)

# Market overview
st.subheader("Market Overview")
left, right = st.columns(2)
with left:
    st.metric("Total Stocks", len(df))
    st.metric("Average Price", f"{df['AvgPrice'].mean():.2f}" if "AvgPrice" in df.columns else "N/A")
with right:
    green_pct = (df['YearlyReturn'] > 0).mean() * 100 if "YearlyReturn" in df.columns else 0
    st.metric("Green Stocks (%)", f"{green_pct:.1f}%")
    st.metric("Avg Volatility", f"{df['Volatility'].mean():.4f}" if "Volatility" in df.columns else "N/A")

# Filtered table
filtered = df[df['YearlyReturn'] >= min_return].copy() if "YearlyReturn" in df.columns else df.copy()
if sector_filter and sector_filter != 'All' and os.path.exists("sector_data.csv"):
    filtered = filtered.merge(pd.read_csv("sector_data.csv"), on='Symbol', how='left')
    filtered = filtered[filtered['Sector'] == sector_filter]

st.subheader("Stock Summary Table")
st.dataframe(filtered.sort_values('YearlyReturn', ascending=False).reset_index(drop=True), height=300)

# Top gainers/losers (year)
st.subheader("Top 10 Gainers and Losers (Yearly)")
col1, col2 = st.columns(2)
with col1:
    st.write("Top 10 Gainers")
    if "YearlyReturn" in df.columns:
        st.dataframe(df.nlargest(10, 'YearlyReturn').reset_index(drop=True))
    else:
        st.info("YearlyReturn column missing.")
with col2:
    st.write("Top 10 Losers")
    if "YearlyReturn" in df.columns:
        st.dataframe(df.nsmallest(10, 'YearlyReturn').reset_index(drop=True))
    else:
        st.info("YearlyReturn column missing.")

# Volatility chart
st.subheader("Top 10 Most Volatile Stocks")
if "Volatility" in df.columns:
    top_vol = df.nlargest(10, "Volatility")
    fig1, ax1 = plt.subplots(figsize=(10,3))
    ax1.bar(top_vol["Symbol"], top_vol["Volatility"])
    ax1.set_xticklabels(top_vol["Symbol"], rotation=45, ha='right')
    ax1.set_ylabel("Volatility")
    st.pyplot(fig1)
else:
    st.info("Volatility data missing.")

# Cumulative returns
st.subheader("Cumulative Returns Over Time")
if os.path.exists(cum_path):
    cum = pd.read_csv(cum_path, index_col=0, parse_dates=True)
    # Ensure numeric
    cum = cum.apply(pd.to_numeric, errors='coerce')
    final_returns = cum.dropna(axis=1, how='all').iloc[-1].sort_values(ascending=False)
    max_k = min(10, len(final_returns))
    if max_k <= 0:
        st.info("Not enough cumulative return data to plot.")
    else:
        k = st.slider("Top K performers to plot", 1, max_k, min(5, max_k))
        top_symbols = final_returns.iloc[:k].index.tolist()
        fig2, ax2 = plt.subplots(figsize=(10,4))
        for sym in top_symbols:
            ax2.plot(cum.index, cum[sym], label=sym)
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Cumulative Return")
        ax2.legend(loc='upper left', bbox_to_anchor=(1,1))
        st.pyplot(fig2)
else:
    st.info("Cumulative returns file not found. Run data_analysis.py")

# Sector-wise performance
st.subheader("Sector-wise Average Yearly Return")
if os.path.exists(sector_path):
    sector_df = pd.read_csv(sector_path)
    fig3, ax3 = plt.subplots(figsize=(10,3))
    ax3.bar(sector_df['Sector'], sector_df['YearlyReturn'])
    ax3.set_xticklabels(sector_df['Sector'], rotation=45, ha='right')
    ax3.set_ylabel("Average Yearly Return")
    st.pyplot(fig3)
else:
    st.info("sector_performance.csv not found. Run data_analysis.py with a sector_data.csv present.")

# Correlation heatmap
st.subheader("Correlation Heatmap (Daily % returns)")
if os.path.exists(corr_path):
    corr = pd.read_csv(corr_path, index_col=0)
    fig4, ax4 = plt.subplots(figsize=(8,6))
    cax = ax4.imshow(corr.values, interpolation='nearest', vmin=-1, vmax=1)
    ax4.set_xticks(np.arange(len(corr.columns)))
    ax4.set_yticks(np.arange(len(corr.index)))
    ax4.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax4.set_yticklabels(corr.index, fontsize=8)
    fig4.colorbar(cax, ax=ax4, fraction=0.046, pad=0.04)
    st.pyplot(fig4)
else:
    st.info("correlation_matrix.csv not found. Run data_analysis.py")

# Monthly top gainers/losers (on-demand)
st.subheader("Monthly Top 5 Gainers & Losers (Compute on demand)")
compute_monthly = st.button("Compute monthly gainers/losers now")
if compute_monthly:
    if os.path.exists(close_matrix_path):
        close = pd.read_csv(close_matrix_path, index_col=0, parse_dates=True)
        # melt to long dataframe
        dfm = close.reset_index().melt(id_vars=['Date'], var_name='Symbol', value_name='Close')
        dfm['Month'] = dfm['Date'].dt.to_period('M')
        grouped = dfm.groupby(['Month','Symbol']).agg(first_close=('Close','first'), last_close=('Close','last')).reset_index()
        grouped['MonthlyReturn'] = (grouped['last_close'] - grouped['first_close']) / grouped['first_close']
        months = sorted(grouped['Month'].unique())
        month_sel = st.selectbox("Pick month", options=[str(m) for m in months])
        if month_sel:
            month_period = pd.Period(month_sel)
            month_df = grouped[grouped['Month'] == month_period]
            st.markdown("**Top 5 Gainers**")
            st.dataframe(month_df.nlargest(5, 'MonthlyReturn')[['Symbol','MonthlyReturn']].reset_index(drop=True))
            st.markdown("**Top 5 Losers**")
            st.dataframe(month_df.nsmallest(5, 'MonthlyReturn')[['Symbol','MonthlyReturn']].reset_index(drop=True))
    else:
        st.info("close_matrix.csv not found. Run data_analysis.py first.")

st.markdown("---")
st.caption("Generated artifacts are stored under `data/` (close_matrix.csv, cumulative_returns.csv, correlation_matrix.csv, sector_performance.csv).")
