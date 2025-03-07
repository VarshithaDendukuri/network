import streamlit as st
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math

G = None

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

def plot_metric_vs_beta(G, initial_infected, beta_values, centrality_measure, L=None):
    lrac_vals = []
    grac_vals = []
    degree_vals = []
    closeness_vals = []
    betweenness_vals = []

    centralities = calculate_centralities(G)

    for beta in beta_values:
        sir_history = sir_model(G, beta, 0.0, initial_infected)
        
        lrac = local_relative_average_centrality(G, initial_infected, L, centrality_measure)
        grac = global_relative_average_centrality(G, initial_infected, centrality_measure)
        degree = centralities['Degree Centrality'][initial_infected]
        closeness = centralities['Closeness Centrality'][initial_infected]
        betweenness = centralities['Betweenness Centrality'][initial_infected]

        lrac_vals.append(lrac)
        grac_vals.append(grac)
        degree_vals.append(degree)
        closeness_vals.append(closeness)
        betweenness_vals.append(betweenness)

    plt.figure(figsize=(15, 10))

    plt.plot(beta_values, lrac_vals, label="LRAC", color='orange')
    plt.plot(beta_values, grac_vals, label="GRAC", color='purple')
    plt.plot(beta_values, degree_vals, label="Degree Centrality", color='blue')
    plt.plot(beta_values, closeness_vals, label="Closeness Centrality", color='green')
    plt.plot(beta_values, betweenness_vals, label="Betweenness Centrality", color='red')

    plt.xlabel("Infection Rate (β)")
    plt.ylabel("Centrality Values")
    plt.title(f"Centrality Measures vs Infection Rate for Node {initial_infected}")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

beta_values = np.linspace(0.01, 1.0, 20)

st.title("📊 Graph Centrality and Epidemic Simulation")

uploaded_file = st.file_uploader("📂 Upload an edge list file", type=["csv", "txt"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, delim_whitespace=True, header=None)
    G = nx.from_pandas_edgelist(df, source=0, target=1)
    
    st.success(f"Graph loaded with **{G.number_of_nodes()}** nodes and **{G.number_of_edges()}** edges.")
    
    if len(list(G.nodes())) > 0:
        initial_infected = random.choice(list(G.nodes()))
        plot_metric_vs_beta(G, initial_infected, beta_values, 'Closeness Centrality', 2)
