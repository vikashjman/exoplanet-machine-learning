import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st 

@st.cache_data
def plot_correlation_heatmap(data):
    plt.figure(figsize=(10, 10))
    corr = data.corr()
    sns.heatmap(corr)
    plt.title('Correlation in the Data')
    plt.tight_layout()
    return plt.gcf()  # Return the current figure
