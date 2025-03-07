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

def top_influential_nodes(G, centrality_measure, top_n=10):
    centralities = calculate_centralities(G)[centrality_measure]
    sorted_nodes = sorted(centralities.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return sorted_nodes

st.title("📊 Graph Centrality and Epidemic Simulation")

uploaded_file = st.file_uploader("📂 Upload an edge list file", type=["csv", "txt"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, delim_whitespace=True, header=None)
    G = nx.from_pandas_edgelist(df, source=0, target=1)
    
    st.success(f"Graph loaded with **{G.number_of_nodes()}** nodes and **{G.number_of_edges()}** edges.")
    
    st.header("🔍 Local Centrality Analysis")
    L = st.slider("Select Level L", 1, 5, 2, key="local_L")
    centrality_measure = st.selectbox("Choose Centrality Measure", list(calculate_centralities(G).keys()), key="local_centrality_measure")
    
    if st.button("Compute Local Centrality", key="compute_local"):
        v = random.choice(list(G.nodes()))
        st.session_state.local_result = local_relative_average_centrality(G, v, L, centrality_measure)
        st.session_state.local_node = v
    
    if st.session_state.get("local_result") is not None:
        st.info(f"Local Relative Average Centrality for node **{st.session_state.local_node}**: **{st.session_state.local_result:.4f}**")
    
    st.header("🌍 Global Centrality Analysis")
    global_node = st.selectbox("Select Node for Global Centrality", list(G.nodes()), key="global_node_select")
    global_centrality_measure = st.selectbox("Choose Centrality Measure", list(calculate_centralities(G).keys()), key="global_centrality_measure")
    
    if st.button("Compute Global Centrality", key="compute_global"):
        st.session_state.global_result = global_relative_average_centrality(G, global_node, global_centrality_measure)
        st.session_state.global_node = global_node
    
    if st.session_state.get("global_result") is not None:
        st.info(f"Global Relative Average Centrality for node **{st.session_state.global_node}**: **{st.session_state.global_result:.4f}**")
    
    st.header("🏆 Top 10 Influential Nodes")
    top_centrality_measure = st.selectbox("Choose Centrality Measure for Top Nodes", list(calculate_centralities(G).keys()), key="top_centrality_measure")
    
    if st.button("Compute Top 10 Influential Nodes", key="compute_top"):
        st.session_state.top_nodes = top_influential_nodes(G, top_centrality_measure)
    
    if st.session_state.get("top_nodes"):
        st.write("### Top 10 Nodes by Centrality (Persistent View)")
        st.table(pd.DataFrame(st.session_state.top_nodes, columns=["Node", "Centrality Value"]))
    
    st.header("🦠 SIR Epidemic Simulation")
    beta = st.slider("Infection Rate (β)", 0.01, 1.0, 0.1, 0.01, key="beta")
    gamma = st.slider("Recovery Rate (γ)", 0.01, 1.0, 0.05, 0.01, key="gamma")
    initial_infected = st.selectbox("Select Initial Infected Node", list(G.nodes()), key="initial_infected")
    
    if st.button("Run SIR Simulation", key="run_sir"):
        history = sir_model(G, beta, gamma, initial_infected)
        plot_sir(history)
def plot_metric_vs_beta(G, initial_infected, gamma, beta_values, centrality_measure, L=None):
    lrac_vals = []
    grac_vals = []
    degree_vals = []
    closeness_vals = []
    betweenness_vals = []

    for beta in beta_values:
        sir_history = sir_model(G, beta, gamma, initial_infected)
        infected_nodes = sir_history[-1][1]  # Number of infected nodes at last step

        lrac = local_relative_average_centrality(G, initial_infected, L, centrality_measure)
        grac = global_relative_average_centrality(G, initial_infected, centrality_measure)
        degree = calculate_centralities(G)['Degree Centrality'][initial_infected]
        closeness = calculate_centralities(G)['Closeness Centrality'][initial_infected]
        betweenness = calculate_centralities(G)['Betweenness Centrality'][initial_infected]

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

