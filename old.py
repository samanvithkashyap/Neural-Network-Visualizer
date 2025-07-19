import streamlit as st
from pyvis.network import Network
import streamlit.components.v1 as components
import torch
import torch.nn as nn
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# --- App Configuration ---
st.set_page_config(page_title="Interactive NN Trainer", layout="wide")


# --- Custom Styling (Optional) ---
def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"{file_name} not found. Using default styling.")

local_css("style.css")


# --- PYTORCH MODEL ---
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation_fn):
        super(SimpleNet, self).__init__()
        layers = []
        
        activations = {"relu": nn.ReLU(), "sigmoid": nn.Sigmoid(), "tanh": nn.Tanh()}
        self.activation = activations[activation_fn]

        layer_sizes = [input_size] + hidden_sizes + [output_size]

        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:
                layers.append(self.activation)

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


# --- DATASET & PLOTTING FUNCTIONS ---
def get_dataset(name, noise):
    if name == "Moons":
        X, y = make_moons(n_samples=200, noise=noise, random_state=42)
    elif name == "Circles":
        X, y = make_circles(n_samples=200, noise=noise, factor=0.5, random_state=42)
    else:
        X, y = make_blobs(n_samples=200, centers=2, n_features=2, cluster_std=noise*10, random_state=42)
    
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    return (torch.FloatTensor(X_train), torch.FloatTensor(y_train).view(-1, 1),
            torch.FloatTensor(X_test), torch.FloatTensor(y_test).view(-1, 1))

def plot_decision_boundary(model, X, y, epoch, accuracy, loss):
    fig, ax = plt.subplots()
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    mesh_tensor = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
    with torch.no_grad():
        Z = torch.sigmoid(model(mesh_tensor)).reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.8)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='k')
    ax.set_title(f"Epoch: {epoch+1}")
    ax.set_xlabel(f"Accuracy: {accuracy:.2f}% | Loss: {loss:.4f}")
    ax.set_xticks(()); ax.set_yticks(())
    return fig


# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("🔬 Data & Model")
    dataset_name = st.selectbox("Select Dataset", ["Moons", "Circles", "Blobs"])
    noise = st.slider("Dataset Noise", 0.0, 0.5, 0.15)
    
    st.header("🧠 Architecture")
    input_size_viz = st.number_input("Input Layer Neurons", min_value=1, max_value=20, value=3)
    num_hidden = st.slider("Number of Hidden Layers", 0, 5, 2)
    hidden_sizes = []
    for i in range(num_hidden):
        size = st.number_input(f"Hidden Layer {i+1}", min_value=1, max_value=20, value=8, key=f"h_{i}")
        hidden_sizes.append(size)
    output_size_viz = st.number_input("Output Layer Neurons", min_value=1, max_value=20, value=1)
    activation = st.selectbox("Activation Function", ["relu", "sigmoid", "tanh"])

    st.header("⚙️ Training")
    learning_rate = st.select_slider("Learning Rate", options=[0.0001, 0.001, 0.01, 0.1, 1.0], value=0.01)
    epochs = st.number_input("Epochs", 10, 5000, 500)
    
    start_training = st.button("🚀 Start Training")

# --- MAIN APP ---
st.title("🧠 Neural Network Trainer & Visualizer")
st.markdown("Build an architecture, visualize it, and train it on sample data.")

tab1, tab2 = st.tabs(["📊 Live Training Dashboard", "🕸️ Architecture Visualizer"])

# --- TAB 1: LIVE TRAINING ---
with tab1:
    if start_training:
        X_train, y_train, X_test, y_test = get_dataset(dataset_name, noise)
        input_size_train = X_train.shape[1]
        output_size_train = 1
        
        model = SimpleNet(input_size_train, hidden_sizes, output_size_train, activation)
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        
        st.header("Live Training Progress")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Decision Boundary")
            plot_placeholder = st.empty()
        with col2:
            st.markdown("#### Metrics Over Time")
            loss_chart = st.line_chart()
            
        losses, accuracies = [], []
        for epoch in range(epochs):
            model.train()
            y_pred = model(X_train)
            loss = loss_fn(y_pred, y_train)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                model.eval()
                with torch.no_grad():
                    test_pred = model(X_test)
                    predicted = (torch.sigmoid(test_pred) > 0.5).float()
                    accuracy = (predicted.eq(y_test).sum() / float(y_test.shape[0])) * 100
                    losses.append(loss.item())
                    loss_chart.add_rows([loss.item()])
                    fig = plot_decision_boundary(model, X_test.numpy(), y_test.numpy().ravel(), epoch, accuracy, loss.item())
                    plot_placeholder.pyplot(fig)
                    plt.close(fig)

        st.success(f"Training finished! Final Accuracy: {accuracy:.2f}%")
    else:
        st.info("Configure your model and press 'Start Training' in the sidebar to begin.")


# --- TAB 2: ARCHITECTURE VISUALIZER ---
with tab2:
    st.header("Interactive Model Architecture")
    physics_enabled = st.toggle("Enable Physics Simulation", value=False, help="Toggle between a fixed layout and a dynamic, interactive one.")
    
    def build_network_graph(input_n, hidden_layers, output_n, activation_fn, physics):
        net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", notebook=True, directed=True)

        if not physics:
            # Disable physics for a static, clean layout
            net.set_options('var options = { "physics": { "enabled": false } }')
        else:
            # Enable physics for an interactive layout
            net.set_options("""
            var options = { "physics": { "barnesHut": { "gravitationalConstant": -30000, "springLength": 250, "springConstant": 0.05 }}}
            """)

        # --- Manual Positioning Algorithm ---
        # This calculates precise X and Y coordinates for a clean, symmetrical layout.
        all_layers = [input_n] + hidden_layers + [output_n]
        total_layers = len(all_layers)
        X_SPACING = 300
        Y_SPACING = 100

        # Add Nodes with calculated positions
        for i, layer_size in enumerate(all_layers):
            layer_x = i * X_SPACING
            # Calculate the starting Y to center the layer vertically
            start_y = -((layer_size - 1) * Y_SPACING) / 2.0
            
            for j in range(layer_size):
                node_id = f"{i}_{j}"
                node_y = start_y + j * Y_SPACING
                
                if i == 0:
                    label, color = f"Input {j+1}", "#00C4A5"
                elif i == total_layers - 1:
                    label, color = f"Output {j+1}", "#4ECDC4"
                else:
                    label, color = f"H{i}_{j+1}", "#FF6B6B"
                
                # For static layout, set fixed positions and disable physics for the node
                if not physics:
                    net.add_node(node_id, label=label, color=color, x=layer_x, y=node_y, physics=False)
                else:
                    net.add_node(node_id, label=label, color=color)

        # Add Edges
        for i in range(total_layers - 1):
            for j in range(all_layers[i]):
                for k in range(all_layers[i+1]):
                    from_node = f"{i}_{j}"
                    to_node = f"{i+1}_{k}"
                    net.add_edge(from_node, to_node)
        
        # Add Activation Node (positioned after the output layer)
        act_x = total_layers * X_SPACING
        net.add_node(
            "activation", label=activation_fn, shape="box", color="#F7B801",
            x=act_x, y=0, physics=not physics
        )
        for k in range(all_layers[-1]):
            net.add_edge(f"{total_layers-1}_{k}", "activation")
            
        return net

    # Build and display the graph using the sidebar controls
    graph = build_network_graph(input_size_viz, hidden_sizes, output_size_viz, activation, physics_enabled)
    
    try:
        path = '/tmp'
        file_path = f'{path}/pyvis_graph.html'
        graph.save_graph(file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            html_data = f.read()
        components.html(html_data, height=800, scrolling=True)
    except Exception as e:
        st.error(f"Error displaying graph: {e}")
