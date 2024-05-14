import pandas as pd
import requests
import streamlit as st
import io

@st.cache_data
def load_data(github_raw_url):
    """Download a file from GitHub given its raw URL."""
    response = requests.get(github_raw_url)
    response.raise_for_status()  # Ensure we got a successful response
    print(response.text)
    data = pd.read_csv(io.StringIO(response.text), error_bad_lines=False)
    return data