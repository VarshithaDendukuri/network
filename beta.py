import streamlit as st
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd

def calculate_centralities(G):
    return {
        'Degree Centrality': nx.degree_centrality(G),
        'Closeness Centrality': nx.closeness_centrality(G),
        'Betweenness Centrality': nx.betweenness_centrality(G),
    }

def sir_model(G, beta, gamma, initial_infected, steps=50):
    infected = {initial_infected}
    recovered = set()
    susceptible = set(G.nodes()) - infected
    
    for _ in range(steps):
        new_infected = set()
        for node in infected:
            for neighbor in G.neighbors(node):
                if neighbor in susceptible and np.random.rand() < beta:
                    new_infected.add(neighbor)
        
        recovered.update(node for node in infected if np.random.rand() < gamma)
        infected = (infected | new_infected) - recovered
        susceptible -= infected
    
    return len(infected)

def plot_centrality_vs_infected(G, beta):
    random_node = random.choice(list(G.nodes()))
    centralities = calculate_centralities(G)

    lrac = sir_model(G, beta, 0.02, random_node)
    grac = sir_model(G, beta, 0.02, random_node)
    degree = centralities['Degree Centrality'][random_node]
    closeness = centralities['Closeness Centrality'][random_node]
    betweenness = centralities['Betweenness Centrality'][random_node]

    results = {
        "LRAC": lrac,
        "GRAC": grac,
        "Degree Centrality": degree,
        "Closeness Centrality": closeness,
        "Betweenness Centrality": betweenness
    }

    plt.figure(figsize=(10, 6))
    bars = plt.bar(results.keys(), results.values(), color='skyblue')
    plt.title(f"Comparison of SIR Model Results for Different Centrality Measures")
    plt.xlabel("Centrality Measure")
    plt.ylabel(f"Number of Infected Nodes (β={beta:.2f})")
    plt.grid(True)

    for bar, value in zip(bars, results.values()):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.2f}', ha='center', va='bottom')

    st.pyplot(plt)

st.title("📊 Centrality vs Infection Rate Visualization")

uploaded_file = st.file_uploader("Upload Edge List File (CSV)", type=["csv", "txt"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, delim_whitespace=True, header=None)
    G = nx.from_pandas_edgelist(df, source=0, target=1)
    st.success(f"Graph loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    beta = st.slider("Select Infection Rate (β)", 0.01, 1.0, 0.1, 0.01)
    
    if st.button("Generate Graph"):
        plot_centrality_vs_infected(G, beta)
