def build_model_config(input_size, hidden_sizes, output_size, activation):
    return {
        "input": input_size,
        "hidden": hidden_sizes,
        "output": output_size,
        "activation": activation
    }