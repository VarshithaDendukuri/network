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

# Function to calculate Local Relative Average Centrality
def local_relative_average_centrality(G, v, L, centrality_measure):
    neighbors = list(nx.single_source_shortest_path_length(G, v, cutoff=L).keys())
    subgraph = G.subgraph(neighbors)
    centrality = calculate_centralities(subgraph)
    return centrality[centrality_measure][v]

# Function to calculate Global Relative Average Centrality
def global_relative_average_centrality(G, v, centrality_measure):
    avg_centrality_G = calculate_centralities(G)
    G_v_removed = G.copy()
    G_v_removed.remove_node(v)
    avg_centrality_G_v = calculate_centralities(G_v_removed)
    return (np.mean(list(avg_centrality_G_v[centrality_measure].values())) - np.mean(list(avg_centrality_G[centrality_measure].values()))) / np.mean(list(avg_centrality_G[centrality_measure].values()))

# SIR Model Function
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

# Function to Plot Bar Graph for Centrality Measures
def plot_centrality_comparison(beta_values, lrac_vals, grac_vals, degree_vals, closeness_vals, betweenness_vals):
    labels = ['LRAC', 'GRAC', 'Degree Centrality', 'Closeness Centrality', 'Betweenness Centrality']
    values = [lrac_vals, grac_vals, degree_vals, closeness_vals, betweenness_vals]

    plt.figure(figsize=(12, 6))
    plt.bar(labels, values, color=['orange', 'purple', 'blue', 'green', 'red'], alpha=0.6)
    plt.xlabel("Centrality Measure")
    plt.ylabel("Number of Infected Nodes")
    plt.title(f"Comparison of SIR Model Results for Different Centrality Measures (Beta={beta_values:.2f})")
    plt.grid(True)
    st.pyplot(plt)

st.title("📊 Graph Centrality and Epidemic Simulation")

uploaded_file = st.file_uploader("📂 Upload an edge list file", type=["csv", "txt"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, delim_whitespace=True, header=None)
    G = nx.from_pandas_edgelist(df, source=0, target=1)

    st.success(f"Graph loaded with **{G.number_of_nodes()}** nodes and **{G.number_of_edges()}** edges.")

    beta = st.slider("Select Beta Value (Infection Rate)", 0.01, 1.0, 0.02, 0.01)
    initial_infected = random.choice(list(G.nodes()))
    centrality_measure = 'Closeness Centrality'
    L = 2

    lrac = local_relative_average_centrality(G, initial_infected, L, centrality_measure)
    grac = global_relative_average_centrality(G, initial_infected, centrality_measure)
    centralities = calculate_centralities(G)

    degree = centralities['Degree Centrality'][initial_infected]
    closeness = centralities['Closeness Centrality'][initial_infected]
    betweenness = centralities['Betweenness Centrality'][initial_infected]

    if st.button("Generate Graph"):
        plot_centrality_comparison(beta, lrac, grac, degree, closeness, betweenness)
