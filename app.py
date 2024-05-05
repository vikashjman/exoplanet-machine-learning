import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from models import (
    load_reports, 
    train_and_evaluate_decision_tree, 
    train_and_evaluate_svm, train_and_evaluate_knn, 
    train_and_evaluate_logistic_regression, 
    train_and_evaluate_naive_bayes, 
    train_and_evaluate_random_forest, 
    train_and_evaluate_rnn
)
import content


import os
# Custom functions for EDA and plotting
# Assuming these functions are defined in your 'eda' module and work as expected
from eda.label_distribution import plot_label_distribution
from eda.flux_variation import plot_flux_variation
from eda.flux_plots import show_graph
from eda.correlation_heatmap import plot_correlation_heatmap

# Function to load data
def load_data(file_path):
    return pd.read_csv(file_path)

# Load datasets
train_data = load_data('data/exoTrain.csv')
test_data = load_data('data/exoTest.csv')

# Page configuration
st.set_page_config(page_title='Exoplanet Detection Analysis Dashboard', layout='wide')

# Page title
st.title('Exoplanet Detection Analysis Dashboard')

# Tab setup
tab1, tab2, tab3 = st.tabs(["Home", "Exploratory Data Analysis", "Model Training and Evaluation"])



# For saving and loading models

# Assuming the path exists, if not you should create it
models_dir = 'models'
reports_dir = os.path.join(models_dir, 'reports')
os.makedirs(models_dir, exist_ok=True)
os.makedirs(reports_dir, exist_ok=True)





with tab1:
    st.header('Exoplanets: The Quest for New Worlds')
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(content.what_are_exoplanet)
        
        # Image placeholder for exoplanet illustration
        try:
            st.image('./images/download.jpeg')
        except Exception as e:
            st.error(f"An error occurred when loading the exoplanet illustration: {e}")
        
    with col2:
        st.write(content.history_of_exoplanet)
        
        # Image placeholder for historical exoplanet discovery
        try:
            st.image('./images/history_exo1.png')
        except Exception as e:
            st.error(f"An error occurred when loading the historical discovery image: {e}")

    st.write(content.ml_section)
    
    # Image placeholders within the ML section
    try:
        st.image('./images/exoplanet.jpg')
    except Exception as e:
        st.error(f"An error occurred when loading the machine learning image: {e}")
        
    # Additional subheadings and paragraphs can be added here as needed.
    st.write(content.about_the_data)
# Implement the other tabs accordingly


with tab2:
    st.header('Exploratory Data Analysis')
    # Label distribution
    st.markdown("""
        ## 🌟 Label Distribution
        Explore the balance between different classes.
    """, unsafe_allow_html=True)

    fig = plot_label_distribution(train_data)
    st.pyplot(fig)
    st.warning(content.imbalance_warning)
    
    st.markdown("""
        <h4 style='text-align: left; color: black;'>🔍 Correlation Heatmap</h4>
        <p>Explore the relationships between different variables.</p>
    """, unsafe_allow_html=True)
    
    fig = plot_correlation_heatmap(train_data)
    st.pyplot(fig, use_container_width=True)  # Adjust the width to fit the container

    st.markdown("""
        <h4 style='text-align: left; color: black;'>📊 Analysis</h4>
        <p>The correlation matrix offers limited insights in this context. As the features represent 
        independent measurements at different times, their interdependence is minimal, 
        providing no significant patterns of correlation.</p>
    """, unsafe_allow_html=True)
    # Flux variation plots
    st.markdown("## 🔭 Flux Variation Insights")
    train_y = train_data[train_data['LABEL'] == 1]
    train_n = train_data[train_data['LABEL'] == 0]
    train_t_y = train_y.iloc[:, 1:].T
    train_t_n = train_n.iloc[:, 1:].T
    fig_flux_var = plot_flux_variation(train_t_n, train_t_y)
    st.plotly_chart(fig_flux_var, use_container_width=True)
    st.markdown("""
    - **Exoplanet Stars**: The flux measurements reveal discernible periodic patterns indicative of planetary transits.
    - **Non-Exoplanet Stars**: The flux appears more constant, with fewer noticeable variations over time.
    - **Observational Anomalies**: Certain irregularities in flux could point to measurement inconsistencies or external interference.
    """)
    
   
# with tab3:
#     st.header('Data Preprocessing')

with tab3:
    c1,c2 = st.columns([8,1])
    
    with c1:
        st.header('Model Training and Evaluation')
    with c2:
        if st.button('Refresh 🔃'):
            # This re-runs the script to update the comparative table
            st.rerun()
    reports = load_reports()
    st.table(pd.DataFrame(reports).T)  # Styling the table

    st.subheader('Select a model to retrain and evaluate:')
    model_options = {
        'SVM': train_and_evaluate_svm,
        'Logistic Regression': train_and_evaluate_logistic_regression,
        'Random Forest': train_and_evaluate_random_forest,
        'K Nearest Neighbours': train_and_evaluate_knn,
        'Decision Tree': train_and_evaluate_decision_tree,
        'Naive Bayes': train_and_evaluate_naive_bayes,
        'RNN': train_and_evaluate_rnn
    }
    model_selection = st.selectbox("Choose a model to train:", options=list(model_options.keys()))

    if st.button('Train and Evaluate 👟'):
        model, report, confusion = model_options[model_selection](train_data, test_data)
        
        # Display success message
        st.success(f'Successfully ran {model_selection} model!')

        # Display reports and confusion matrix in two columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f'#### {model_selection} Model Classification Report')
            st.dataframe(pd.DataFrame(report).T)  # Transform the report dictionary to a DataFrame and transpose it for better viewing

        with col2:
            st.write(f'#### {model_selection} Confusion Matrix')
            fig, ax = plt.subplots(figsize=(8, 6))  # Sizing the plot to make it more readable
            sns.heatmap(confusion, annot=True, fmt='d', ax=ax, cmap='Blues')
            st.pyplot(fig)

        if st.button('Refresh 🗒️'):
            # This re-runs the script to update the comparative table
            st.rerun()
