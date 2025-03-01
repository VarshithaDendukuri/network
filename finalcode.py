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

def global_relative_average_centrality(G, v):
    centrality = calculate_centralities(G)
    return {key: centrality[key][v] for key in centrality}

def local_relative_average_centrality(G, v, L, centrality_measure):
    neighbors = list(nx.single_source_shortest_path_length(G, v, cutoff=L).keys())
    subgraph = G.subgraph(neighbors)
    centrality = calculate_centralities(subgraph)
    return centrality[centrality_measure][v]

def sir_model(G, beta, gamma, steps=10):
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

st.title("Network Analysis Tool")

uploaded_file = st.file_uploader("Upload Graph (Edge List CSV or TXT)", type=["csv", "txt"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, delim_whitespace=True, header=None)
    G = nx.from_pandas_edgelist(df, source=0, target=1)
    
    st.header("Global Centrality Analysis")
    global_centralities = {v: global_relative_average_centrality(G, v) for v in G.nodes()}
    df_global = pd.DataFrame.from_dict(global_centralities, orient='index')
    st.dataframe(df_global)
    
    st.subheader("Top 10 Influencing Nodes")
    st.dataframe(df_global.nlargest(10, 'degree_centrality'))
    
    st.header("Local Centrality Analysis")
    L = st.slider("Select Level L", 1, 5, 2)
    centrality_measure = st.selectbox("Choose Centrality Measure", list(calculate_centralities(G).keys()))
    if st.button("Compute Local Centrality"):
        local_centralities = {v: local_relative_average_centrality(G, v, L, centrality_measure) for v in G.nodes()}
        df_local = pd.DataFrame.from_dict(local_centralities, orient='index', columns=[centrality_measure])
        st.dataframe(df_local)
    
    st.header("SIR Model Simulation")
    beta = st.slider("Infection Rate (Beta)", 0.0, 1.0, 0.2)
    gamma = st.slider("Recovery Rate (Gamma)", 0.0, 1.0, 0.1)
    if st.button("Run SIR Model"):
        sir_results = sir_model(G, beta, gamma)
        plot_sir(sir_results)

st.write("Upload a network graph as a CSV or TXT file with two columns representing edges.")
