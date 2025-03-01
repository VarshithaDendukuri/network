import streamlit as st
import networkx as nx
import pandas as pd
import random
import matplotlib.pyplot as plt
import numpy as np

# Function to compute global centralities
def compute_global_centralities(G):
    centralities = {
        'Degree Centrality': nx.degree_centrality(G),
        'Closeness Centrality': nx.closeness_centrality(G),
        'Betweenness Centrality': nx.betweenness_centrality(G, seed=random.choice(list(G.nodes))),
        'Eigenvector Centrality': nx.eigenvector_centrality(G),
        'Clustering Coefficient': nx.clustering(G),
        'Katz Centrality': nx.katz_centrality_numpy(G, 1/1.1 - 0.01)
    }
    return centralities

# Function to compute local centralities
def compute_local_centralities(G, node, L):
    neighbors = set([node])
    for _ in range(L):
        level_neighbors = set()
        for n in neighbors:
            level_neighbors.update(G.neighbors(n))
        neighbors.update(level_neighbors)
    
    H = G.subgraph(neighbors)
    local_centralities = {
        'Degree Centrality': nx.degree_centrality(H),
        'Closeness Centrality': nx.closeness_centrality(H),
        'Betweenness Centrality': nx.betweenness_centrality(H, seed=random.choice(list(H.nodes))),
        'Eigenvector Centrality': nx.eigenvector_centrality(H),
        'Clustering Coefficient': nx.clustering(H),
        'Katz Centrality': nx.katz_centrality_numpy(H, 1/1.1 - 0.01)
    }
    return local_centralities

# SIR Model Simulation
def sir_model(G, beta, gamma, initial_infected, steps):
    states = {node: 'S' for node in G.nodes()}
    for node in initial_infected:
        states[node] = 'I'
    
    results = []
    for _ in range(steps):
        new_states = states.copy()
        for node in G.nodes():
            if states[node] == 'I':
                if random.random() < gamma:
                    new_states[node] = 'R'
                else:
                    for neighbor in G.neighbors(node):
                        if states[neighbor] == 'S' and random.random() < beta:
                            new_states[neighbor] = 'I'
        states = new_states.copy()
        results.append(states.copy())
    return results

# Streamlit UI
st.title("Network Centrality Analysis with SIR Model")

# File uploader
uploaded_file = st.file_uploader("Upload an edge list CSV or TXT file", type=["csv", "txt"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file, delim_whitespace=True, header=None, names=["Source", "Target"])
    
    G = nx.from_pandas_edgelist(df, source=df.columns[0], target=df.columns[1])
    
    # Global Centrality Analysis
    if st.button("Run Global Centrality Analysis"):
        global_centralities = compute_global_centralities(G)
        for key, values in global_centralities.items():
            st.subheader(key)
            sorted_values = sorted(values.items(), key=lambda x: x[1], reverse=True)
            st.dataframe(pd.DataFrame(sorted_values, columns=['Node', key]))
    
    # Local Centrality Analysis
    node = st.text_input("Enter a node for local analysis:")
    L = st.slider("Select neighborhood level (L):", 1, 5, 2)
    
    if st.button("Run Local Centrality Analysis") and node:
        if node in G.nodes:
            local_centralities = compute_local_centralities(G, node, L)
            for key, values in local_centralities.items():
                st.subheader(f"Local {key}")
                sorted_values = sorted(values.items(), key=lambda x: x[1], reverse=True)
                st.dataframe(pd.DataFrame(sorted_values, columns=['Node', key]))
        else:
            st.error("Node not found in the graph!")
    
    # Top 10 Influencing Nodes
    st.subheader("Top 10 Influencing Nodes")
    degree_centrality = nx.degree_centrality(G)
    top_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
    st.dataframe(pd.DataFrame(top_nodes, columns=['Node', 'Degree Centrality']))
    
    # Graph Visualization
    st.subheader("Graph Visualization")
    plt.figure(figsize=(8, 6))
    nx.draw(G, with_labels=True, node_color='skyblue', edge_color='gray')
    st.pyplot(plt)
    
    # SIR Model Simulation
    st.subheader("SIR Model Simulation")
    beta = st.slider("Infection Rate (β)", 0.01, 1.0, 0.1)
    gamma = st.slider("Recovery Rate (γ)", 0.01, 1.0, 0.1)
    initial_infected = st.multiselect("Select Initial Infected Nodes", list(G.nodes()))
    steps = st.slider("Number of Simulation Steps", 1, 50, 10)
    
    if st.button("Run SIR Simulation"):
        if initial_infected:
            sir_results = sir_model(G, beta, gamma, initial_infected, steps)
            infected_counts = [sum(1 for state in states.values() if state == 'I') for states in sir_results]
            
            st.line_chart(infected_counts)
        else:
            st.error("Please select at least one initial infected node.")
