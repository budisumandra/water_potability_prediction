import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import plotly.express as px
from matplotlib.patches import Rectangle


def visualization_missingvalue(df):
    colors = ['#DFF6FF', '#47B5FF', '#256D85', "#06283D"]
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    nan_data_numerical = round(
        100*(df.isna().sum())/(len(df.index)), 2).sort_values(ascending=False).to_frame()
    fig, axs = plt.subplots(1, 1, figsize=(10, 7.5))
    plt1 = sns.heatmap(nan_data_numerical, annot=True, cmap=colors, ax=axs)
    #axs.set_title('Numerical Columns\n',fontweight = 'bold',fontsize=15)
    plt.suptitle('MISSING VALUES \n PER COLUMN\n',
                 fontsize=20, fontweight='bold')
    plt.tight_layout()
    st.write('\n')
    st.pyplot(fig)


def visualization_comparison(df):
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    colors = ['#DFF6FF', '#47B5FF', '#256D85', "#06283D"]
    nan_data_numerical = round(
        100*(df.isna().sum())/(len(df.index)), 2).sort_values(ascending=False).to_frame()
    data = df.copy()
    list_nan_features = list(
        nan_data_numerical[nan_data_numerical[0] > 0].index)
    for col in list_nan_features:
        data[col] = data[col].replace(np.nan, data[col].median())
    fig = plt.figure(figsize=(10, 6))
    ax = sns.countplot(data['potability'],
                       order=data['potability'].value_counts().index)

    # Create annotate
    for i in ax.patches:
        ax.text(x=i.get_x()+i.get_width()/2, y=i.get_height()/7, s=f"{np.round(i.get_height()/len(data)*100,0)}%",
                ha='center', size=50, weight='bold', rotation=90, color='white')
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.0f'),
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center',
                    va='center',
                    xytext=(0, 10),
                    textcoords='offset points')

    plt.title("Comparison of Potable and Not Potable Samples \n",
              size=15, weight='bold')
    plt.annotate(text="Not safe water for human consumption", xytext=(0.5, 1790), xy=(0.2, 1250),
                 arrowprops=dict(arrowstyle="->", color='blue', connectionstyle="angle3,angleA=0,angleB=90"), color='black')
    plt.annotate(text="Safe water for human consumption", xytext=(0.8, 1600), xy=(1.2, 1000),
                 arrowprops=dict(arrowstyle="->", color='blue',  connectionstyle="angle3,angleA=0,angleB=90"), color='black')

    # Setting Plot
    sns.despine(right=True, top=True, left=True)
    ax.axes.yaxis.set_visible(False)
    st.pyplot(fig)


def visualization_distribution(df):
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    colors = ['#DFF6FF', '#47B5FF', '#256D85', "#06283D"]
    nan_data_numerical = round(
        100*(df.isna().sum())/(len(df.index)), 2).sort_values(ascending=False).to_frame()
    data = df.copy()
    list_nan_features = list(
        nan_data_numerical[nan_data_numerical[0] > 0].index)
    for col in list_nan_features:
        data[col] = data[col].replace(np.nan, data[col].median())

    cols = data.columns[0:9].to_list()
    min_val = [6.5, 60, 500, 0, 3, 200, 0, 0, 0]
    max_val = [8.5, 120, 1000, 4, 250, 400, 10, 80, 5]
    limit = pd.DataFrame(data=[min_val, max_val], columns=col)

    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(
        15, 15), constrained_layout=True)
    plt.suptitle(
        'Feature distribution by Potability class and Approved limit', size=20, weight='bold')
    ax = ax.flatten()
    for x, i in enumerate(cols):
        sns.kdeplot(data=data, x=i, hue='potability', ax=ax[x], fill=True, multiple='stack', alpha=0.5,
                    linewidth=0)
        l, k = limit.iloc[:, x]
        print(ax[x].add_patch(Rectangle(xy=(l, 0), width=k-l, height=1, alpha=0.5)))
        for s in ['left', 'right', 'top', 'bottom']:
            ax[x].spines[s].set_visible(False)
    print(fig.show())

    # st.pyplot(fig)
