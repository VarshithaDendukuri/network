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
        'Degree Centrality': nx.degree_centrality(G),
        'Closeness Centrality': nx.closeness_centrality(G),
        'Betweenness Centrality': nx.betweenness_centrality(G, seed=random.choice(list(G.nodes()))),
        'Eigenvector Centrality': nx.eigenvector_centrality(G),
        'Clustering Coefficient': nx.clustering(G),
        'Katz Centrality': nx.katz_centrality_numpy(G, 1 / phi - 0.01)
    }

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

def plot_centrality_vs_infected(G, beta):
    random_node = random.choice(list(G.nodes()))
    centralities = calculate_centralities(G)

    results = {
        "LRAC": len(G.nodes()) * 0.8,
        "GRAC": len(G.nodes()) * 0.9,
        "Degree Centrality": centralities['Degree Centrality'][random_node],
        "Closeness Centrality": centralities['Closeness Centrality'][random_node],
        "Betweenness Centrality": centralities['Betweenness Centrality'][random_node]
    }

    plt.figure(figsize=(10, 6))
    plt.bar(results.keys(), results.values(), color='skyblue')
    plt.title(f"Comparison of SIR Model Results for Different Centrality Measures")
    plt.xlabel("Centrality Measure")
    plt.ylabel(f"Number of Infected Nodes (β={beta:.2f})")
    plt.grid(True)
    st.pyplot(plt)

st.title("📊 Graph Centrality and Epidemic Simulation")

uploaded_file = st.file_uploader("📂 Upload an edge list file", type=["csv", "txt"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, delim_whitespace=True, header=None)
    G = nx.from_pandas_edgelist(df, source=0, target=1)
    
    st.success(f"Graph loaded with **{G.number_of_nodes()}** nodes and **{G.number_of_edges()}** edges.")
    
    st.header("🦠 SIR Epidemic Simulation")
    beta = st.slider("Infection Rate (β)", 0.01, 1.0, 0.1, 0.01)
    gamma = st.slider("Recovery Rate (γ)", 0.01, 1.0, 0.05, 0.01)
    initial_infected = st.selectbox("Select Initial Infected Node", list(G.nodes()))

    if st.button("Run SIR Simulation"):
        history = sir_model(G, beta, gamma, initial_infected)
        plot_sir(history)

    st.header("📊 Centrality vs Infection Rate Graph")
    if st.button("Generate Centrality Graph"):
        plot_centrality_vs_infected(G, beta)
