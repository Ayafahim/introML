
## S19, Q10) Which one of the curves in Figure 5 will then correspond to the function f ?

![[Pasted image 20240515214200.png]]

## Solution

#### Neural Network Description:
The problem statement might describe the neural network in terms of:
1. **Layers**:
   - Mention of the number of layers, such as "one hidden layer" and "one output layer," explicitly states the architecture. If not explicitly stated, you might infer it from the context or the formulas given.
2. **Weights**:
   - **$w^{(1)}$**: Weights connecting the input layer to the first hidden layer.
   - **$w^{(2)}$**: Weights connecting the last hidden layer to the output layer.
3. **Activation Functions**:
   - Mention of specific functions applied at each layer, such as sigmoid in the hidden layer and a linear (or identity) function in the output layer, further helps delineate the structure.

#### Identifying Hidden and Output Layers:
- **Hidden Layer(s)**:
  - Typically processes inputs via weighted connections followed by an activation function (e.g., ReLU, sigmoid).
  - In the context of the given question, if the formula includes a transformation like $h^{(1)}([1, x_1, x_2] w_j^{(1)})$, you can infer that $h^{(1)}$ applies to the hidden layer's output. The weights $w^{(1)}$ are thus associated with the hidden layer.
- **Output Layer**:
  - The final layer that produces the output of the network.
  - If a linear transformation such as $w_0^{(2)} + \sum_{j=1}^2 w_j^{(2)} h^{(1)}(\ldots)$ is described, and $h^{(2)}$ is mentioned as linear or identity, this is characteristic of an output layer in a regression context or a simple network configuration.

#### Practical Example:
In the ANN configuration described:
- **One Hidden Layer** with two neurons (implied by the mention of two sets of weights $w^{(1)}_1$ and $w^{(1)}_2$).
- **One Output Layer** with a single neuron (implied by the use of a single linear transformation combining the outputs of the hidden layer).


Use the script below and try entering some different points and see if the value matches the plots given. It also provides a plot for you. Remember to switch out the activation functions you need. 2 different scripts.

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, x * alpha)

def prelu(x, alpha=0.25):
    return np.where(x > 0, x, x * alpha)

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def tanh(x):
    return np.tanh(x)

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # stability improvement by subtracting max
    return exp_x / exp_x.sum(axis=0, keepdims=True)

def swish(x, beta=1.0):
    return x * sigmoid(beta * x)

def mish(x):
    return x * np.tanh(np.log1p(np.exp(x)))

def linear(x):
    return x

def neural_network_output(x1, x2, w, activation_hidden=sigmoid, activation_output=linear):
    w1 = np.array(w['w1'])
    w2 = np.array(w['w2'])
    w0 = np.array(w['w0'])
    w02 = w['w2_0']

    h1 = activation_hidden(np.dot([1, x1, x2], w1))
    h2 = activation_hidden(np.dot([1, x1, x2], w2))

    output = activation_output(w02 + w0[1] * h1 + w0[2] * h2)
    return output

def plot_ann_outputs(weights, x1_range, x2_range, activation_hidden, activation_output):
    outputs = []
    x1_vals, x2_vals = np.meshgrid(x1_range, x2_range)

    for x1, x2 in zip(x1_vals.flatten(), x2_vals.flatten()):
        output = neural_network_output(x1, x2, weights, activation_hidden, activation_output)
        outputs.append(output)

    outputs = np.array(outputs).reshape(x1_vals.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    c = ax.pcolormesh(x1_vals, x2_vals, outputs, cmap='viridis')
    fig.colorbar(c, ax=ax)
    plt.title('ANN Output Landscape')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.show()

# Define weight structure
weights = {
    'w1': [-1.2, -1.3, 0.6],
    'w2': [-1.0, 0.0, 0.9],
    'w0': [0, 0.5, 0.5],  # Output layer weights
    'w2_0': 2.2  # Output layer bias
}

# Define ranges for inputs
x1_range = np.linspace(-1, 1, 100)
x2_range = np.linspace(-1, 1, 100)

# Choose activation functions
activation_hidden = relu  # Change to sigmoid, relu, tanh, etc.
activation_output = linear  # Change to linear, softmax, etc., for output layer specifics

# Plot the outputs
plot_ann_outputs(weights, x1_range, x2_range, activation_hidden, activation_output)

```



```python
import numpy as np  
import matplotlib.pyplot as plt  
  
  
def sigmoid(x):  
    return 1 / (1 + np.exp(-x))  
  
def neural_network_output(x1, x2, w):  
    # Unpack weights  
    w1 = np.array(w['w1'])  
    w2 = np.array(w['w2'])  
    w0 = w['w0']  
    w02 = w['w2_0']  
  
    # Compute output from the first hidden layer  
    h1 = sigmoid(np.dot([1, x1, x2], w1))  
    h2 = sigmoid(np.dot([1, x1, x2], w2))  
  
    # Compute the final output  
    output = w02 + w0[1] * h1 + w0[2] * h2  
    return output  
  
# Weights from the question  
weights = {  
    'w1': [-1.2, -1.3, 0.6],  
    'w2': [-1.0, -0.0, 0.9],  
    'w0': [0, 0.5, 0.5],  # Assuming these based on typical structure  
    'w2_0': 2.2  
}  
  
# Example calculation  
# You can vary x1 and x2 to plot and see which graph it matches  
output = neural_network_output(0.5, 0.5, weights)  
print("Output of the neural network for inputs (0.5, 0.5):", output)  
  
  
def plot_ann_outputs(weights, x1_range, x2_range):  
    outputs = []  
    x1_vals, x2_vals = np.meshgrid(x1_range, x2_range)  
  
    for x1, x2 in zip(x1_vals.flatten(), x2_vals.flatten()):  
        output = neural_network_output(x1, x2, weights)  
        outputs.append(output)  
  
    outputs = np.array(outputs).reshape(x1_vals.shape)  
  
    fig, ax = plt.subplots(figsize=(8, 6))  
    c = ax.pcolormesh(x1_vals, x2_vals, outputs, cmap='viridis')  
    fig.colorbar(c, ax=ax)  
    plt.title('ANN Output Landscape')  
    plt.xlabel('$x_1$')  
    plt.ylabel('$x_2$')  
    plt.show()  
  
  
# Define weight structure from your specific network description  
weights = {  
    'w1': [-1.2, -1.3, 0.6],  
    'w2': [-1.0, -0.0, 0.9],  
    'w0': [0, 0.5, 0.5],  # Assuming these based on typical structure  
    'w2_0': 2.2  
}  
  
# Define ranges for inputs  
x1_range = np.linspace(-1, 1, 100)  
x2_range = np.linspace(-1, 1, 100)  
  
# Plot the outputs  
plot_ann_outputs(weights, x1_range, x2_range)
```


