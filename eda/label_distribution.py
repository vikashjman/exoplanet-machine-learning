import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import streamlit as st

@st.cache_data
def plot_label_distribution(data):
    # Set a style for the plots
    sns.set_style("whitegrid")
    
    # Adjust the size for a more subtle appearance
    fig, ax = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [2, 1]})
    
    # Use a more attractive color palette and adjust plot properties
    palette = sns.color_palette("husl", 2)
    
    # Count plot for label distribution
    sns.countplot(x='LABEL', data=data, palette=palette, ax=ax[0])
    ax[0].set_title('Count of Exoplanet vs Non-Exoplanet Stars', fontsize=14)
    ax[0].set_xlabel('Label', fontsize=12)
    ax[0].set_ylabel('Count', fontsize=12)
    ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))  # Ensures integer ticks on x-axis

    # Pie chart for label proportion
    data['LABEL'].value_counts().plot.pie(
        explode=[0, 0.1], autopct='%1.1f%%', ax=ax[1], shadow=True, colors=palette,
        startangle=90, pctdistance=0.85, labels=['Non-Exoplanet', 'Exoplanet']
    )
    ax[1].set_title('Proportion of Exoplanet vs Non-Exoplanet Stars', fontsize=14)
    ax[1].set_ylabel('')  # Remove the 'Label' ylabel

    # Draw circle for a more aesthetically pleasing pie chart (donut chart)
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig.gca().add_artist(centre_circle)
    
    # Ensure tight layout and return the figure object
    plt.tight_layout()
    return fig
