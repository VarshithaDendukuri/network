import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

# Data from the table
data = {
    "Beta": [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.15, 0.20, 0.30],
    "LRAC": [299, 378, 420, 425, 435, 443, 446, 439, 449, 452],
    "GRAC": [312, 392, 411, 423, 434, 437, 445, 435, 449, 452],
    "Degree Centrality": [186, 298, 325, 373, 373, 397, 407, 403, 406, 439],
    "Closeness Centrality": [186, 291, 321, 376, 377, 394, 412, 409, 415, 434],
    "Betweenness Centrality": [175, 288, 325, 376, 368, 393, 399, 409, 418, 433]
}

# Convert data into DataFrame
df = pd.DataFrame(data)

st.title("Centrality Measures vs Infection Rate")

# Upload dataset option
uploaded_file = st.file_uploader("Upload your dataset", type=["csv,txt"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset uploaded successfully!")

# Slider for Beta
beta = st.slider("Select Beta Value", min_value=0.02, max_value=0.30, step=0.02)

# Filter data according to the selected Beta value
filtered_df = df[df["Beta"] == beta]

if not filtered_df.empty:
    plt.figure(figsize=(10, 6))
    plt.bar(["LRAC", "GRAC", "Degree Centrality", "Closeness Centrality", "Betweenness Centrality"],
            filtered_df.iloc[0, 1:], color='skyblue')
    plt.title(f"Comparison of SIR Model Results for Beta = {beta}")
    plt.xlabel("Centrality Measure")
    plt.ylabel("Number of Infected Nodes")
    plt.grid(True)
    st.pyplot(plt)
else:
    st.warning("No data available for the selected Beta value")
