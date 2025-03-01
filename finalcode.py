import streamlit as st
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math

def calculate_centralities(G):
    phi = (1 + math.sqrt(5)) / 2.0
    return {
        'degree_centrality': nx.degree_centrality(G),
        'closeness_centrality': nx.closeness_centrality(G),
        'betweenness_centrality': nx.betweenness_centrality(G, seed=random.choice(list(G.nodes))),
        'eigenvector_centrality': nx.eigenvector_centrality(G),
        'clustering_coefficient': nx.clustering(G),
        'katz_centrality': nx.katz_centrality_numpy(G, 1 / phi - 0.01)
    }

def global_relative_average_centrality(G, v, centrality_measure):
    avg_centrality_G = calculate_centralities(G)
    G_v_removed = G.copy()
    G_v_removed.remove_node(v)
    avg_centrality_G_v = calculate_centralities(G_v_removed)
    return (np.mean(list(avg_centrality_G_v[centrality_measure].values())) - np.mean(list(avg_centrality_G[centrality_measure].values()))) / np.mean(list(avg_centrality_G[centrality_measure].values()))

def local_relative_average_centrality(G, v, L, centrality_measure):
    neighbors = list(nx.single_source_shortest_path_length(G, v, cutoff=L).keys())
    subgraph = G.subgraph(neighbors)
    centrality = calculate_centralities(subgraph)
    return centrality[centrality_measure][v]

def sir_model(G, beta, gamma, steps=50):
    infected = set(np.random.choice(list(G.nodes()), size=1))
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
        susceptible = susceptible - infected
        sir_results.append((len(susceptible), len(infected), len(recovered)))
    
    return sir_results

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

st.title("Graph Centrality and Epidemic Simulation")

uploaded_file = st.file_uploader("Upload an edge list file", type=["csv", "txt"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, delim_whitespace=True, header=None)
    G = nx.from_pandas_edgelist(df, source=0, target=1)
    
    st.write(f"Graph loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    
    st.header("Local Centrality Analysis")
    L = st.slider("Select Level L", 1, 5, 2)
    centrality_measure = st.selectbox("Choose Centrality Measure", list(calculate_centralities(G).keys()))
    
    if st.button("Compute Local Centrality"):
        v = random.choice(list(G.nodes()))
        local_result = local_relative_average_centrality(G, v, L, centrality_measure)
        st.write(f"Local Relative Average Centrality for node {v}: {local_result}")
    
    st.header("Global Centrality Analysis")
    global_node = st.selectbox("Select Node for Global Centrality", list(G.nodes()))
    global_centrality_measure = st.selectbox("Choose Centrality Measure", list(calculate_centralities(G).keys()))
    
    if st.button("Compute Global Centrality"):
        global_result = global_relative_average_centrality(G, global_node, global_centrality_measure)
        st.write(f"Global Relative Average Centrality for node {global_node}: {global_result}")
    
    st.header("SIR Epidemic Simulation")
    beta = st.slider("Infection Rate (β)", 0.01, 1.0, 0.1, 0.01)
    gamma = st.slider("Recovery Rate (γ)", 0.01, 1.0, 0.05, 0.01)
    initial_infected = st.selectbox("Select Initial Infected Node", list(G.nodes()))
    
    if st.button("Run SIR Simulation"):
        history = sir_model(G, beta, gamma, initial_infected)
        plot_sir(history)
