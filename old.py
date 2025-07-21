import streamlit as st
from pyvis.network import Network
import streamlit.components.v1 as components
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import tempfile
import os

# --- App Configuration ---
st.set_page_config(page_title="Interactive TF/Keras Trainer", layout="wide")


# --- Custom Styling (Optional) ---
def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"'{file_name}' not found. Using default styling.")

local_css("style.css")


# --- TENSORFLOW/KERAS MODEL ---
def create_model(input_size, hidden_sizes, output_size, activation_fn):
    """Builds a TensorFlow Keras Sequential model."""
    model = Sequential()
    
    # Define the first layer (input and first hidden)
    if hidden_sizes:
        model.add(Dense(hidden_sizes[0], activation=activation_fn, input_shape=(input_size,)))
        # Add remaining hidden layers
        for size in hidden_sizes[1:]:
            model.add(Dense(size, activation=activation_fn))
    else: # Handle case with no hidden layers
        model.add(Dense(output_size, input_shape=(input_size,)))
            
    # Add the output layer if there were hidden layers
    if hidden_sizes:
        model.add(Dense(output_size)) # No activation, as we use from_logits=True in the loss
    
    return model


# --- DATASET & PLOTTING FUNCTIONS ---
def get_dataset(name, noise):
    """Generates and prepares the selected dataset."""
    if name == "Moons":
        X, y = make_moons(n_samples=200, noise=noise, random_state=42)
    elif name == "Circles":
        X, y = make_circles(n_samples=200, noise=noise, factor=0.5, random_state=42)
    else: # Blobs
        X, y = make_blobs(n_samples=200, centers=2, n_features=2, cluster_std=noise * 10, random_state=42)
    
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    return (tf.constant(X_train, dtype=tf.float32), 
            tf.constant(y_train.reshape(-1, 1), dtype=tf.float32),
            tf.constant(X_test, dtype=tf.float32), 
            tf.constant(y_test.reshape(-1, 1), dtype=tf.float32))

def plot_decision_boundary(model, X, y, epoch, accuracy, loss):
    """Plots the decision boundary of the model."""
    fig, ax = plt.subplots()
    x_min, x_max = X[:, 0].numpy().min() - 0.5, X[:, 0].numpy().max() + 0.5
    y_min, y_max = X[:, 1].numpy().min() - 0.5, X[:, 1].numpy().max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    mesh_tensor = tf.constant(np.c_[xx.ravel(), yy.ravel()], dtype=tf.float32)
    
    Z = tf.sigmoid(model(mesh_tensor)).numpy().reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.8)
    ax.scatter(X[:, 0], X[:, 1], c=y.numpy().ravel(), cmap=plt.cm.RdYlBu, edgecolors='k')
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
    # Note: Input size is determined by the data, so the UI input is for visualization only.
    input_size_viz = st.number_input("Input Layer Neurons", min_value=2, max_value=20, value=2, disabled=True)
    num_hidden = st.slider("Number of Hidden Layers", 0, 5, 2)
    hidden_sizes = []
    for i in range(num_hidden):
        size = st.number_input(f"Hidden Layer {i+1}", min_value=1, max_value=20, value=8, key=f"h_{i}")
        hidden_sizes.append(size)
    output_size_viz = st.number_input("Output Layer Neurons", min_value=1, max_value=20, value=1, disabled=True)
    activation = st.selectbox("Activation Function", ["relu", "sigmoid", "tanh"])

    st.header("⚙️ Training")
    learning_rate = st.select_slider("Learning Rate", options=[0.0001, 0.001, 0.01, 0.1, 1.0], value=0.01)
    epochs = st.number_input("Epochs", 10, 5000, 500)
    
    start_training = st.button("🚀 Start Training")

# --- MAIN APP ---
st.title("🧠 Neural Network Trainer")
st.markdown("Build an architecture, visualize it, and train it on sample data.")

tab1, tab2 = st.tabs(["📊 Live Training Dashboard", "🕸️ Architecture Visualizer"])

# --- TAB 1: LIVE TRAINING ---
with tab1:
    if start_training:
        X_train, y_train, X_test, y_test = get_dataset(dataset_name, noise)
        input_size_train = X_train.shape[1]
        output_size_train = 1
        
        model = create_model(input_size_train, hidden_sizes, output_size_train, activation)
        loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        optimizer = SGD(learning_rate=learning_rate)
        
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
            # Training step
            with tf.GradientTape() as tape:
                y_pred = model(X_train, training=True)
                loss = loss_fn(y_train, y_pred)
            
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            if (epoch + 1) % 10 == 0:
                # Evaluation step
                test_pred_logits = model(X_test, training=False)
                test_loss = loss_fn(y_test, test_pred_logits)
                
                predicted = tf.cast(tf.sigmoid(test_pred_logits) > 0.5, tf.float32)
                accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y_test), tf.float32)) * 100
                
                losses.append(test_loss.numpy())
                loss_chart.add_rows([test_loss.numpy()])
                
                fig = plot_decision_boundary(model, X_test, y_test, epoch, accuracy.numpy(), test_loss.numpy())
                plot_placeholder.pyplot(fig)
                plt.close(fig)

        st.success(f"Training finished! Final Accuracy: {accuracy.numpy():.2f}%")
    else:
        st.info("Configure your model and press 'Start Training' in the sidebar to begin.")


# --- TAB 2: ARCHITECTURE VISUALIZER ---
with tab2:
    st.header("Interactive Model Architecture")
    physics_enabled = st.toggle("Enable Physics Simulation", value=False, help="Toggle between a fixed layout and a dynamic, interactive one.")
    
    def build_network_graph(input_n, hidden_layers, output_n, physics):
        net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", notebook=True, directed=True)

        if not physics:
            net.set_options('var options = { "physics": { "enabled": false } }')
        else:
            net.set_options("""
            var options = { "physics": { "barnesHut": { "gravitationalConstant": -30000, "springLength": 250, "springConstant": 0.05 }}}
            """)

        all_layers = [input_n] + hidden_layers + [output_n]
        total_layers = len(all_layers)
        X_SPACING = 300
        Y_SPACING = 100

        for i, layer_size in enumerate(all_layers):
            layer_x = i * X_SPACING
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
                
                if not physics:
                    net.add_node(node_id, label=label, color=color, x=layer_x, y=node_y, physics=False)
                else:
                    net.add_node(node_id, label=label, color=color)

        for i in range(total_layers - 1):
            for j in range(all_layers[i]):
                for k in range(all_layers[i+1]):
                    from_node = f"{i}_{j}"
                    to_node = f"{i+1}_{k}"
                    net.add_edge(from_node, to_node)
        
        return net

    # Build and display the graph using the sidebar controls
    # We use '2' for input viz because the datasets are 2D, and '1' for the single output node
    graph = build_network_graph(2, hidden_sizes, 1, physics_enabled)
    
    try:
        # Get the system's temporary directory path in a cross-platform way
        temp_dir = tempfile.gettempdir()
        
        # Create a full, OS-compatible path to the file
        file_path = os.path.join(temp_dir, 'pyvis_graph.html')
        
        graph.save_graph(file_path)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            html_data = f.read()
            
        components.html(html_data, height=800, scrolling=True)
        
    except Exception as e:
        st.error(f"Error displaying graph: {e}")
