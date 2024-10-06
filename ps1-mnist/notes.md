# Example Walkthrough: Model Setup and Training Process

## Model Setup (Before Training)

### Initial Setup
We define the model and computation graph first, as done earlier. Let's assume:
- Each image has **784 features** (for MNIST).
- There are **10 possible classes** (digits 0-9).
- One hidden layer with **64 units**.

```python
nInputs = 784  # Number of input features
nHiddens = 64  # Number of hidden units
nLabels = 10   # Number of output classes
```

# Define the model parameters (randomly initialized weights and biases)
```python
Phi = MLPparams(nInputs, nHiddens, nLabels)
```

# Define input and label placeholders
```python
xnode = edf.Input()  # Input for the data (batch of images)
ynode = edf.Input()  # Input for the labels (batch of true labels)
```

# Build the computation graph (forward pass through the network)
```python
probnode = MLPsigmoidgraph(Phi, xnode)  # Network outputs (softmax probabilities)
```

# Loss node (log loss between predicted probabilities and true labels)
```python
lossnode = edf.LogLoss(probnode, ynode)  # Loss to minimize
```

## Example Data (One Batch)

Let’s say we have a batch of 2 images, and we'll use this to illustrate how the network works:

### Example batch of images (`xnode.value`):
```python
xnode.value = np.array([
    [0.5, 0.2, 0.1, ..., 0.8],  # Image 1 with 784 features
    [0.9, 0.4, 0.6, ..., 0.3]   # Image 2 with 784 features
])  # Shape (2, 784)
```

### True labels (`ynode.value`):
Let's say the true labels for these images are:
```python
ynode.value = np.array([3, 7])  # Image 1 is class 3, Image 2 is class 7
```

## Forward Pass
Now we perform the forward pass to compute the output probabilities and the loss. The weights in `Phi` are randomly initialized, so the network will produce some initial probabilities.
```python
edf.Forward()  # Perform the forward pass
```

### Predicted Probabilities
Let’s say after the forward pass, we get the following predicted probabilities from the network:
```python
# Probabilities for 2 images across 10 classes
print(probnode.value)  
# Output (probnode.value) might look like this:
# [[0.05, 0.10, 0.05, 0.50, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],  # Image 1
#  [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.80, 0.02, 0.02]]  # Image 2
```

Here:

- For **Image 1**, the network predicted class **3** with a probability of **0.50**.
- For **Image 2**, the network predicted class **7** with a probability of **0.80**.

### Compute Loss
Now, we compute the log loss for this batch using the predicted probabilities and the true labels:
```python
print(lossnode.value)  
# Output lossnode.value might look like:
# 0.223 (this is the initial loss before training)
```

### Backward Pass and Parameter Update
Next, we perform the backward pass to compute the gradients and update the model parameters:
```python
edf.Backward(lossnode)  # Compute gradients for the loss
edf.SGD()  # Update parameters using stochastic gradient descent
```

At this point, the weights in **Phi** (the parameters) are updated based on the computed gradients.

### Summary of the Process
- **Phi** contains the initial model parameters (weights and biases) which are used during the forward pass.
- **probnode** computes the output probabilities using these parameters for a given batch of input data (**xnode.value**).
- **lossnode** computes the loss between the predicted output (**probnode.value**) and the true labels (**ynode.value**).

During training, the forward pass computes the output and loss, and the backward pass updates the parameters to minimize the loss. Each time you process a batch, only the input (**xnode.value**) and labels (**ynode.value**) change, while the structure of the computation graph (i.e., **probnode** and **lossnode**) remains the same.


