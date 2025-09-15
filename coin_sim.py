import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import binom

# 日本語フォントを指定（WindowsならMS Gothic、Macならヒラギノなど）
rcParams['font.family'] = 'MS Gothic'  # Windowsの場合
# rcParams['font.family'] = 'Hiragino Maru Gothic Pro'  # Macの場合
# rcParams['font.family'] = 'IPAPGothic'  # Linuxの場合（IPAフォント導入時）

st.title("コイン投げシミュレーション（統計的な推測）")

# ユーザー入力
p = st.slider("表が出る確率（%）", 0, 100, 50) / 100
n = st.number_input("投げる回数（1試行あたり）", min_value=1, max_value=10000, value=100)
trials = st.number_input("試行回数（繰り返しシミュレーション数）", min_value=1, max_value=1000, value=50)

if st.button("シミュレーション開始"):
    # ---- シミュレーション実行 ----
    sim_results = []
    for _ in range(trials):
        results = np.random.binomial(n=n, p=p)
        sim_results.append(results)

    sim_results = np.array(sim_results)

    # ---- 1回の試行の結果 ----
    st.subheader("例：1回の試行の結果")
    one_trial = np.random.choice(["表", "裏"], size=n, p=[p, 1-p])
    counts = pd.Series(one_trial).value_counts()

    st.table(counts)
    fig, ax = plt.subplots()
    counts.plot(kind="bar", ax=ax, color=["skyblue", "lightcoral"])
    ax.set_ylabel("回数")
    ax.set_xlabel("結果")
    st.pyplot(fig)

    # ---- 複数試行の分布 ----
    st.subheader(f"{trials}回のシミュレーションによる分布（表の出た回数）")
    fig, ax = plt.subplots()
    ax.hist(sim_results, bins=20, color="skyblue", edgecolor="black")
    ax.set_xlabel("表の出た回数")
    ax.set_ylabel("頻度")
    st.pyplot(fig)

    # ---- 理論値との比較（二項分布） ----
    st.subheader("理論値（二項分布）との比較")
    x = np.arange(0, n+1)
    pmf = binom.pmf(x, n, p)

    fig, ax = plt.subplots()
    ax.hist(sim_results, bins=20, density=True, alpha=0.6, color="skyblue", label="シミュレーション")
    ax.plot(x, pmf, 'r-', lw=2, label="理論（二項分布）")
    ax.set_xlabel("表の出た回数")
    ax.set_ylabel("確率密度")
    ax.legend()
    st.pyplot(fig)

    # ---- 理論値の期待値・分散 ----
    st.subheader("理論値（期待値と分散）")
    expected = n * p
    variance = n * p * (1 - p)
    st.write(f"期待値: {expected:.2f}")
    st.write(f"分散: {variance:.2f}")
