import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

@st.cache_data
def plot_flux_variation(train_t_n, train_t_y):
    # Fixed indices for the stars to plot
    star_indices = [ 37, 5086, 3000, 3001]

    # Create the subplot titles based on the fixed indices
    subplot_titles = [f"Flux Variation of Star {i}" for i in star_indices]

    # Determine the number of rows and columns for the subplots
    num_plots = len(star_indices)
    rows = num_plots // 2 + num_plots % 2
    cols = 2 if num_plots > 1 else 1
    
    # Create subplots
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles)

    # Add traces for the specified stars
    for i, star_index in enumerate(star_indices):
        row = (i // cols) + 1
        col = (i % cols) + 1
        
        if star_index in train_t_n.columns:
            trace_n = go.Scatter(y=train_t_n[star_index], x=train_t_n.index, name=f'Star {star_index} Non-Exoplanet')
            fig.add_trace(trace_n, row=row, col=col)
        elif star_index in train_t_y.columns:
            trace_y = go.Scatter(y=train_t_y[star_index], x=train_t_y.index, name=f'Star {star_index} Exoplanet')
            fig.add_trace(trace_y, row=row, col=col)
    
    # Update the layout of the figure
    fig.update_layout(height=400 * rows, width=800, title_text="Flux Variations Across Stars", showlegend=False)

    return fig
