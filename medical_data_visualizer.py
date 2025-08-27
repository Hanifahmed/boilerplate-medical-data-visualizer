import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ---------- PARAMETERS ----------
CSV_PATH = "medical_examination.csv"
# --------------------------------

def _load_and_prepare_df(path=CSV_PATH):
    """Load CSV and preprocess dataset"""
    df = pd.read_csv(path)

    # BMI and overweight column
    height_m = df['height'] / 100
    df['overweight'] = ((df['weight'] / (height_m ** 2)) > 25).astype(int)

    # Normalize cholesterol and gluc
    df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
    df['gluc'] = (df['gluc'] > 1).astype(int)

    # Ensure binary variables are int
    for col in ['smoke', 'alco', 'active', 'cardio']:
        df[col] = df[col].astype(int)

    return df

def draw_cat_plot():
    """Draw categorical plot"""
    df = _load_and_prepare_df(CSV_PATH)

    df_cat = pd.melt(
        df,
        id_vars=['cardio'],
        value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight']
    )

    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')

    sns.set_theme(style="whitegrid")
    catplot = sns.catplot(
        data=df_cat,
        kind='bar',
        x='variable',
        y='total',
        hue='value',
        col='cardio',
        height=5,
        aspect=1
    )

    catplot.set_axis_labels("variable", "total")
    catplot.set_titles("cardio = {col_name}")
    catplot._legend.set_title("value")

    for ax in catplot.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(30)

    return catplot.fig

def draw_heat_map():
    """Draw correlation heatmap"""
    df = _load_and_prepare_df(CSV_PATH)

    # Clean data according to boilerplate instructions
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    corr = df_heat.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".1f",
        center=0,
        vmax=0.3,
        vmin=-0.1,
        square=True,
        linewidths=.5,
        cbar_kws={"shrink": 0.5},
        ax=ax
    )

    fig.tight_layout()
    return fig

if __name__ == "__main__":
    fig1 = draw_cat_plot()
    fig1.savefig("catplot.png")
    print("Saved catplot.png")

    fig2 = draw_heat_map()
    fig2.savefig("heatmap.png")
    print("Saved heatmap.png")
