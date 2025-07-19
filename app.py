import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network
import torch
import torch.nn as nn
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import os
import tempfile

# --- App Configuration ---
st.set_page_config(page_title="Live NN Visualizer", layout="wide")

# --- Custom CSS for a cleaner look ---
st.markdown("""
<style>
    .stApp {
        background-color: #1E1E1E;
    }
    .stExpander {
        border: 1px solid #444;
        border-radius: 10px;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        background-color: #00C4A5;
        color: white;
        border: none;
        font-weight: bold;
    }
    .legend {
        padding: 10px;
        border: 1px solid #555;
        border-radius: 10px;
        background-color: #2B2B2B;
    }
    .legend-item { display: flex; align-items: center; margin-bottom: 5px; }
    .legend-line { width: 25px; height: 5px; margin-right: 10px; border-radius: 2px; }
    .graph-container {
        width: 100% !important;
        height: 700px !important;
        min-height: 700px !important;
        overflow: auto;
    }
    .network-canvas {
        width: 100% !visportant;
        height: 100% !important;
    }
</style>
""", unsafe_allow_html=True)

# --- PyTorch Model ---
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation_fn):
        super(SimpleNet, self).__init__()
        self.layers = nn.ModuleList()
        activations = {"relu": nn.ReLU(), "sigmoid": nn.Sigmoid(), "tanh": nn.Tanh()}
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:
                self.layers.append(activations[activation_fn])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# --- Dataset & Plotting ---
def get_dataset(name, noise):
    if name == "Moons": 
        X, y = make_moons(n_samples=300, noise=noise, random_state=42)
    elif name == "Circles": 
        X, y = make_circles(n_samples=300, noise=noise, factor=0.5, random_state=42)
    else: 
        X, y = make_blobs(n_samples=300, centers=2, n_features=2, cluster_std=noise*10, random_state=42)
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    return (torch.FloatTensor(X_train), torch.FloatTensor(y_train).view(-1, 1),
            torch.FloatTensor(X_test), torch.FloatTensor(y_test).view(-1, 1))

def plot_decision_boundary(model, X, y):
    fig, ax = plt.subplots(figsize=(6, 5.5))
    x_min, x_max = X[:, 0].min()-0.5, X[:, 0].max()+0.5
    y_min, y_max = X[:, 1].min()-0.5, X[:, 1].max()+0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    with torch.no_grad():
        Z = torch.sigmoid(model(torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]))).reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap='viridis', alpha=0.8)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k')
    ax.set_xticks(()); ax.set_yticks(())
    return fig

# --- Legend & Live Graph Functions ---
def render_legend():
    legend_html = """
    <div class="legend">
        <strong>Hover over graph for exact values.</strong>
        <hr style="margin: 5px 0; border-color: #444;">
        <div class="legend-item"><div class="legend-line" style="background: linear-gradient(90deg, #3498DB, #E64B35);"></div> Weight (Blue: Neg, Red: Pos)</div>
        <div class="legend-item"><div class="legend-line" style="background: linear-gradient(90deg, #A6D9D4, #D5A6BD);"></div> Bias (Teal: Neg, Pink: Pos)</div>
        <small><i>Line thickness indicates weight magnitude.</i></small>
    </div>"""
    st.markdown(legend_html, unsafe_allow_html=True)

def create_live_graph(model):
    net = Network(height="100%", width="100%", notebook=False, directed=True, bgcolor="#2B2B2B", font_color="white")
    net.set_options("""
    var options = {
        "physics": { "enabled": false },
        "nodes": { "shape": "dot", "font": { "color": "white" }, "scaling": { "min": 15, "max": 30 } },
        "edges": { "smooth": { "type": "continuous" } },
        "layout": { "improvedLayout": true, "randomSeed": 42 },
        "width": "100%",
        "height": "100%"
    }
    """)
    linear_layers = [l for l in model.layers if isinstance(l, nn.Linear)]
    all_layer_sizes = [l.in_features for l in linear_layers] + [linear_layers[-1].out_features]
    
    with torch.no_grad():
        max_abs_weight = max(torch.max(torch.abs(l.weight)).item() for l in linear_layers) + 1e-6
        max_abs_bias = max(torch.max(torch.abs(l.bias)).item() for l in linear_layers) + 1e-6

    X_SPACING, Y_SPACING = 250, 100
    for i, layer_size in enumerate(all_layer_sizes):
        layer_x = i * X_SPACING
        start_y = -((layer_size - 1) * Y_SPACING) / 2.0
        for j in range(layer_size):
            node_id, node_y = f"{i}_{j}", start_y + j * Y_SPACING
            if i > 0:
                bias_val = linear_layers[i-1].bias.data[j].item()
                norm = mcolors.Normalize(vmin=-max_abs_bias, vmax=max_abs_bias)
                cmap = plt.cm.PiYG
                color = mcolors.to_hex(cmap(norm(bias_val)))
                title = f"Bias: {bias_val:.4f}"
            else:
                color, title = "#00C4A5", "Input Neuron"
            net.add_node(node_id, color=color, size=15, x=layer_x, y=node_y, label=" ", title=title)

    for i, layer in enumerate(linear_layers):
        for j in range(layer.out_features):
            for k in range(layer.in_features):
                weight = layer.weight.data[j, k].item()
                norm = mcolors.Normalize(vmin=-max_abs_weight, vmax=max_abs_weight)
                cmap = plt.cm.coolwarm
                color = mcolors.to_hex(cmap(norm(weight)))
                width = 0.5 + 4 * (abs(weight) / max_abs_weight)
                title = f"Weight: {weight:.4f}"
                net.add_edge(f"{i}_{k}", f"{i+1}_{j}", color=color, width=width, title=title)
    return net

# --- Sidebar Controls ---
with st.sidebar:
    st.title("🧠 Live Trainer")
    st.markdown("Configure your model, then watch it learn.")
    with st.expander("🔬 Data Source", expanded=True):
        dataset_name = st.selectbox("Select Dataset", ["Moons", "Circles", "Blobs"])
        noise = st.slider("Dataset Noise", 0.0, 0.5, 0.15)
    with st.expander("🏗️ Architecture", expanded=True):
        num_hidden = st.slider("Hidden Layers", 0, 4, 1)
        hidden_sizes = [st.number_input(f"Neurons in Hidden {i+1}", 1, 20, 5, key=f"h_{i}") for i in range(num_hidden)]
        activation = st.selectbox("Activation", ["relu", "sigmoid", "tanh"])
    with st.expander("⚙️ Hyperparameters", expanded=True):
        learning_rate = st.select_slider("Learning Rate", [0.001, 0.01, 0.1, 1.0], 0.01)
        epochs = st.number_input("Epochs", 100, 10000, 1000)
    start_training = st.button("🚀 Start Live Training")

# --- Main App Layout ---
st.title("Live Neural Network Visualization")
st.markdown("The network graph is the **real model**. Hover over any node or link to see its live value.")

col1, col2 = st.columns([1.5, 1])

with col1:
    graph_placeholder = st.empty()
with col2:
    plot_placeholder = st.empty()
    loss_placeholder = st.empty()
    legend_placeholder = st.empty()
    
# Set initial state
with graph_placeholder.container():
    st.info("The live network graph will appear here after you start training.")
with plot_placeholder.container():
    st.info("The decision boundary will appear here.")
with legend_placeholder.container():
    render_legend()

# --- Training Logic ---
if start_training:
    X_train, y_train, X_test, y_test = get_dataset(dataset_name, noise)
    input_size, output_size = X_train.shape[1], 1
    model = SimpleNet(input_size, hidden_sizes, output_size, activation)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    loss_placeholder.empty()
    loss_chart = loss_placeholder.line_chart({"Loss": []})

    with tempfile.TemporaryDirectory() as tmpdirname:
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            y_pred = model(X_train)
            loss = loss_fn(y_pred, y_train)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 50 == 0 or epoch == 0:
                model.eval()
                net = create_live_graph(model)
                try:
                    file_path = os.path.join(tmpdirname, 'live_nn.html')
                    net.save_graph(file_path)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        html_data = f.read()
                    # Inject custom CSS into the HTML to enforce sizing
                    html_data = html_data.replace('<body>', '<body><div class="graph-container"><div class="network-canvas">')
                    html_data = html_data.replace('</body>', '</div></div></body>')
                    
                    with graph_placeholder.container():
                        components.html(html_data, height=700, width=None, scrolling=True)
                
                except Exception as e:
                    with graph_placeholder.container():
                        st.error(f"Could not render graph: {e}")

                loss_chart.add_rows({"Loss": [loss.item()]})
                fig = plot_decision_boundary(model, X_test.numpy(), y_test.numpy().ravel())
                plot_placeholder.pyplot(fig)
                plt.close(fig)
    
    st.success("✔️ Training Complete!")