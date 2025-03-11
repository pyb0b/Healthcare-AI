import torch
from neural_network import NeuralNetwork


def extract_weights_biases(model1):
    """Extracts and prints the weights and biases from each layer of the neural network."""
    for name, param in model1.named_parameters():
        if "weight" in name:
            print(f"Layer: {name}")
            print(f"Weights: {param.data.numpy()}")
            bias_name = name.replace("weight", "bias")
            if bias_name in model1.state_dict():
                print(f"Biases: {model1.state_dict()[bias_name].numpy()}")
            else:
                print("Biases: None")
            print("-" * 50)


# Example usage
if __name__ == "__main__":
    input_size = 8
    model1 = NeuralNetwork(input_size)
    model1.load_state_dict(torch.load("model1.pth"))
    model1.eval()

    extract_weights_biases(model1)

    model2 = NeuralNetwork(input_size)
    model2.load_state_dict(torch.load("model2.pth"))
    model2.eval()

    extract_weights_biases(model2)

    modelf = NeuralNetwork(input_size)
    modelf.load_state_dict(torch.load("federated_model.pth"))
    modelf.eval()

    extract_weights_biases(modelf)
