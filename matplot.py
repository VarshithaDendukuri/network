import streamlit as st
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math

# Function to calculate centralities
def calculate_centralities(G):
    phi = (1 + math.sqrt(5)) / 2.0
    return {
        'Degree Centrality': nx.degree_centrality(G),
        'Closeness Centrality': nx.closeness_centrality(G),
        'Betweenness Centrality': nx.betweenness_centrality(G, seed=random.choice(list(G.nodes))),
        'Eigenvector Centrality': nx.eigenvector_centrality(G),
        'Clustering Coefficient': nx.clustering(G),
        'Katz Centrality': nx.katz_centrality_numpy(G, 1 / phi - 0.01)
    }

# Global Relative Average Centrality
def global_relative_average_centrality(G, v, centrality_measure):
    avg_centrality_G = calculate_centralities(G)
    G_v_removed = G.copy()
    G_v_removed.remove_node(v)
    avg_centrality_G_v = calculate_centralities(G_v_removed)
    return (np.mean(list(avg_centrality_G_v[centrality_measure].values())) - np.mean(list(avg_centrality_G[centrality_measure].values()))) / np.mean(list(avg_centrality_G[centrality_measure].values()))

# Local Relative Average Centrality
def local_relative_average_centrality(G, v, L, centrality_measure):
    neighbors = list(nx.single_source_shortest_path_length(G, v, cutoff=L).keys())
    subgraph = G.subgraph(neighbors)
    centrality = calculate_centralities(subgraph)
    return centrality[centrality_measure][v]

# SIR Model Simulation
def sir_model(G, beta, gamma, initial_infected, steps=50):
    infected = {initial_infected}
    recovered = set()
    susceptible = set(G.nodes()) - infected

    sir_results = []
    for _ in range(steps):
        new_infected = set()
        for node in infected:
            for neighbor in G.neighbors(node):
                if neighbor in susceptible and np.random.rand() < beta:
                    new_infected.add(neighbor)
        
        recovered.update(node for node in infected if np.random.rand() < gamma)
        infected = (infected | new_infected) - recovered
        susceptible -= infected
        sir_results.append((len(susceptible), len(infected), len(recovered)))
    
    return sir_results

# Plot SIR Model Results
def plot_sir(history):
    S_vals, I_vals, R_vals = zip(*history)
    plt.figure(figsize=(10, 5))
    plt.plot(S_vals, label='Susceptible', color='blue')
    plt.plot(I_vals, label='Infected', color='red')
    plt.plot(R_vals, label='Recovered', color='green')
    plt.xlabel("Time Steps")
    plt.ylabel("Number of Nodes")
    plt.legend()
    st.pyplot(plt)

# Top Influential Nodes
def top_influential_nodes(G, centrality_measure, top_n=10):
    centralities = calculate_centralities(G)[centrality_measure]
    sorted_nodes = sorted(centralities.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return sorted_nodes

# Main Application
st.title("📊 Graph Centrality and Epidemic Simulation")

uploaded_file = st.file_uploader("📂 Upload an edge list file", type=["csv", "txt"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, delim_whitespace=True, header=None)
    G = nx.from_pandas_edgelist(df, source=0, target=1)
    st.success(f"Graph loaded with **{G.number_of_nodes()}** nodes and **{G.number_of_edges()}** edges.")

    st.header("🔍 Local Centrality Analysis")
    L = st.slider("Select Level L", 1, 5, 2)
    centrality_measure = st.selectbox("Choose Centrality Measure", list(calculate_centralities(G).keys()))

    if st.button("Compute Local Centrality"):
        v = random.choice(list(G.nodes()))
        result = local_relative_average_centrality(G, v, L, centrality_measure)
        st.info(f"Local Relative Average Centrality for node {v}: {result:.4f}")

    st.header("🌍 Global Centrality Analysis")
    node = st.selectbox("Select Node for Global Centrality", list(G.nodes()))
    centrality_measure_global = st.selectbox("Choose Centrality Measure", list(calculate_centralities(G).keys()), key="global")

    if st.button("Compute Global Centrality"):
        result_global = global_relative_average_centrality(G, node, centrality_measure_global)
        st.info(f"Global Relative Average Centrality for node {node}: {result_global:.4f}")

    st.header("🏆 Top 10 Influential Nodes")
    top_measure = st.selectbox("Choose Centrality Measure for Top Nodes", list(calculate_centralities(G).keys()), key="top")

    if st.button("Compute Top 10 Nodes"):
        top_nodes = top_influential_nodes(G, top_measure)
        st.table(pd.DataFrame(top_nodes, columns=["Node", "Centrality Value"]))

    st.header("🦠 SIR Epidemic Simulation")
    beta = st.slider("Infection Rate (β)", 0.01, 1.0, 0.1)
    gamma = st.slider("Recovery Rate (γ)", 0.01, 1.0, 0.05)
    initial_infected = st.selectbox("Select Initial Infected Node", list(G.nodes()))

    if st.button("Run SIR Simulation"):
        history = sir_model(G, beta, gamma, initial_infected)
        plot_sir(history)

    st.header("Comparison for sir model and centrality measures")
    data = {
        "Beta": [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.15, 0.20, 0.30],
        "LRAC": [299, 378, 420, 425, 435, 443, 446, 439, 449, 452],
        "GRAC": [312, 392, 411, 423, 434, 437, 445, 435, 449, 452],
        "Degree Centrality": [186, 298, 325, 373, 373, 397, 407, 403, 406, 439],
        "Closeness Centrality": [186, 291, 321, 376, 377, 394, 412, 409, 415, 434],
        "Betweenness Centrality": [175, 288, 325, 376, 368, 393, 399, 409, 418, 433]
    }

    df = pd.DataFrame(data)
    beta = st.slider("Select Beta Value", min_value=0.02, max_value=0.30, step=0.02)
    filtered_df = df[df["Beta"] == beta]

    if not filtered_df.empty:
        plt.figure(figsize=(10, 6))
        plt.bar(df.columns[1:], filtered_df.iloc[0, 1:], color='skyblue')
        plt.title(f"Comparison of SIR Model Results for Beta = {beta}")
        plt.xlabel("Centrality Measure")
        plt.ylabel("Number of Infected Nodes")
        plt.grid(True)
        st.pyplot(plt)
    else:
        st.warning("No data available for the selected Beta value")
