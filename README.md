# 📊 Graph Centrality and Epidemic Simulation using Streamlit

## 🚀 Overview
This project is a **Streamlit**-based web application for analyzing graph centrality and simulating epidemic spread using the **SIR model**. It allows users to:
- Upload graph data (edge lists)
- Analyze **local** and **global** centrality measures
- Identify **top 10 influential nodes**
- Simulate the **SIR (Susceptible-Infected-Recovered) model** on a given graph

## 🎯 Features
- **Graph Upload & Visualization**: Load graph data and visualize key statistics.
- **Local & Global Centrality Analysis**: Compute various centrality measures for nodes.
- **Top 10 Influential Nodes**: Identify the most influential nodes based on centrality.
- **SIR Epidemic Simulation**: Model disease spread dynamics and visualize results interactively.
- **Persistent View**: Top nodes and centrality results remain visible even when running the SIR model.

## 🏗️ Tech Stack
- **Python** (Core logic)
- **Streamlit** (Frontend UI)
- **NetworkX** (Graph computations)
- **Matplotlib** (Plotting results)
- **Pandas** (Data handling)
- **NumPy** (Scientific computations)

## 📦 Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```
2. **Create a virtual environment** (Optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the application**:
   ```bash
   streamlit run finalcode.py
   ```

## 📂 Usage
1. Upload a graph edge list file (`.csv` or `.txt`).
2. Choose centrality measures and compute:
   - **Local Centrality Analysis**
   - **Global Centrality Analysis**
3. Get **Top 10 Influential Nodes**.
4. Run **SIR Model Simulation** and visualize epidemic spread.

## 📸 Screenshots
![Graph Centrality UI](https://via.placeholder.com/800x400.png?text=Graph+Centrality+Analysis)

## 🤝 Contributing
Contributions are welcome! Feel free to **fork** the repo, create a **pull request**, or open an **issue** for feature suggestions.

## 📜 License
This project is **MIT Licensed**. Feel free to use and modify it.

---
💡 _Developed with passion using Streamlit & NetworkX!_ 🚀

