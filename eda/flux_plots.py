# flux_plots.py
import matplotlib.pyplot as plt
import streamlit as st

def flux_graph(dataset, row_index, title):
    fig, ax = plt.subplots(figsize=(20, 5))
    ax.set_title(title, color='black', fontsize=22)
    ax.set_xlabel('Time', color='black', fontsize=18)
    ax.set_ylabel('Flux', color='black', fontsize=18)
    ax.grid(True)

    flux_values = dataset.iloc[row_index]
    ax.plot(range(1, len(flux_values) + 1), flux_values, 'black')

    ax.tick_params(colors='black', labelcolor='black', labelsize=14)
    st.pyplot(fig)  # This line will handle plotting directly in Streamlit.

def show_graph(dataframe, dataset):
    with_planet_indices = dataframe[dataframe['LABEL'] == 2].head(3).index
    wo_planet_indices = dataframe[dataframe['LABEL'] == 1].head(3).index

    for index in with_planet_indices:
        flux_graph(dataset, index, "Periodic dip due to transiting planet")
    
    for index in wo_planet_indices:
        flux_graph(dataset, index, "No transiting planet")
