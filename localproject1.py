import streamlit as st
import networkx as nx
import pandas as pd
import numpy as np
import random
import math
import scipy
import matplotlib.pyplot as plt

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

def average_centrality(G):
    centralities = calculate_centralities(G)
    return {measure: np.mean(list(values.values())) for measure, values in centralities.items()}

def global_relative_average_centrality(G, v):
    avg_centrality_G = average_centrality(G)
    G_v_removed = G.copy()
    G_v_removed.remove_node(v)
    avg_centrality_G_v = average_centrality(G_v_removed)
    return {key: (avg_centrality_G_v[key] - avg_centrality_G[key]) / avg_centrality_G[key] for key in avg_centrality_G}

def local_relative_average_centrality(G, v, L, centrality_measure):
    neighbors = set([v])
    for _ in range(L):
        level_neighbors = set()
        for node in neighbors:
            level_neighbors.update(G.neighbors(node))
        neighbors.update(level_neighbors)
    
    H = G.subgraph(neighbors)
    average_centrality_H = sum(calculate_centralities(H)[centrality_measure].values()) / len(H)
    
    neighbors_without_v = neighbors - {v}
    H_without_v = G.subgraph(neighbors_without_v)
    average_centrality_H_without_v = sum(calculate_centralities(H_without_v)[centrality_measure].values()) / len(H_without_v)
    
    return (average_centrality_H_without_v - average_centrality_H) / average_centrality_H

def sir_simulation(G, beta, gamma, initial_infected, steps=50):
    S = set(G.nodes()) - {initial_infected}
    I = {initial_infected}
    R = set()
    
    history = []
    for _ in range(steps):
        new_infected = set()
        for node in I:
            for neighbor in G.neighbors(node):
                if neighbor in S and random.random() < beta:
                    new_infected.add(neighbor)
        
        recovered = {node for node in I if random.random() < gamma}
        
        S -= new_infected
        I = (I | new_infected) - recovered
        R |= recovered
        
        history.append((len(S), len(I), len(R)))
    
    return history

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

def main():
    st.title("Graph Centrality and Epidemic Simulation")
    uploaded_file = st.file_uploader("Upload an edge list file", type=["txt", "csv"])
    
    if uploaded_file is not None:
        G = nx.Graph()
        df = pd.read_csv(uploaded_file, delim_whitespace=True, header=None)
        for _, row in df.iterrows():
            G.add_edge(int(row[0]) - 1, int(row[1]) - 1)
        
        st.write(f"Graph loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
        
        st.header("Local Centrality Analysis")
        v = random.choice(list(G.nodes))
        L = st.slider("Select Level L", min_value=1, max_value=5, value=2)
        centrality_measure = st.selectbox("Choose Centrality Measure", ['degree_centrality', 'closeness_centrality', 'betweenness_centrality', 'eigenvector_centrality', 'katz_centrality'])
        
        if st.button("Compute Local Centrality"):
            local_result = local_relative_average_centrality(G, v, L, centrality_measure)
            st.write(f"Local Relative Average Centrality for node {v}: {local_result}")
        
        st.header("Global Centrality Analysis")
        if st.button("Compute Global Centralities"):
            global_centralities = {v: global_relative_average_centrality(G, v) for v in G.nodes()}
            df_centralities = pd.DataFrame.from_dict(global_centralities, orient='index')
            st.dataframe(df_centralities)
        
        st.header("SIR Epidemic Simulation")
        beta = st.slider("Infection Rate (β)", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
        gamma = st.slider("Recovery Rate (γ)", min_value=0.01, max_value=1.0, value=0.05, step=0.01)
        initial_infected = st.selectbox("Select Initial Infected Node", list(G.nodes()))
        
        if st.button("Run SIR Simulation"):
            history = sir_simulation(G, beta, gamma, initial_infected)
            plot_sir(history)

if __name__ == "__main__":
    main()
