# 1. **Our Explanation of the Training Process**:
- **Initialize random weights** (no bias assumed).
- Perform a **forward pass** for one training example through the network (2 hidden layers).
- Compute the loss and **backpropagate** to calculate gradients for each layer’s weights:  
  $$ \left[ \frac{\partial \text{Loss}}{\partial W_1}, \frac{\partial \text{Loss}}{\partial W_2}, \frac{\partial \text{Loss}}{\partial W_3} \right] $$
- **Update the weights** using **Stochastic Gradient Descent (SGD)**:  
  $$ W_1 = W_1 - \eta \frac{\partial \text{Loss}}{\partial W_1}, \quad W_2 = W_2 - \eta \frac{\partial \text{Loss}}{\partial W_2}, \quad W_3 = W_3 - \eta \frac{\partial \text{Loss}}{\partial W_3} $$
- Repeat this for the next training example, using the updated weights from the previous example.
- After one pass through all training examples (one **epoch**), we end up with updated weights.
- Typically, **multiple epochs** are run to optimize the weights.

---

# 2. **Professor's Code Breakdown**:
- **Forward pass** (`Forward()`): Iterates over computational nodes to process input.
- **Backward pass** (`Backward(loss)`):
  - Resets gradients.
  - Initializes the gradient of the loss function (`loss.grad = 1.`).
  - Backpropagates through the computational nodes in reverse order.
- **SGD (Stochastic Gradient Descent):** Updates each parameter by subtracting the learning rate times the gradient.

This matches **our understanding** of SGD: forward pass → backpropagation → weight update.

---

# **Understanding the Blog Post (Batch Learning)**

Blog post: https://machinelearningmastery.com/neural-networks-crash-course/

Now, the blog post introduces **batch learning** as an alternative to SGD, so let’s explain how that works:

- **Stochastic Gradient Descent (SGD):**
   This is what we’ve already described and what the professor’s code does—one training example at a time is passed through the network, the loss is calculated, gradients are computed, and the weights are updated immediately after each example. This is called **online learning**, and it tends to update the model frequently, but can sometimes make the training a bit chaotic (especially if there’s noise in the data).

- **Batch Learning:**
   Instead of updating the weights after every single training example, batch learning waits until it has processed all the training examples (or a "mini-batch" of them). Here’s how it works:

   1. **Forward pass for each training example:** 
      We process all the training examples, compute the outputs, and collect the loss for each one. But we don’t update the weights yet.

   2. **Backpropagation:** 
      After processing the entire batch (or mini-batch), we compute the **average gradient** for each layer over all examples in the batch.  
      If there are $N$ examples in a batch, the average gradient for a layer $W_1$ would be:  
      $$ \frac{1}{N} \sum_{i=1}^{N} \frac{\partial \text{Loss}_i}{\partial W_1} $$
      where $\frac{\partial \text{Loss}_i}{\partial W_1}$ is the gradient for the $i$-th example in the batch.

   3. **Weight Update:**
      After averaging the gradients over the batch, we update the weights:  
      $$ W_1 = W_1 - \eta \cdot \frac{1}{N} \sum_{i=1}^{N} \frac{\partial \text{Loss}_i}{\partial W_1} $$
      So, instead of using the gradient from a single example, we use the **averaged gradient** over the whole batch. This can make training more stable, but each update is slower since we’re waiting for a full batch before updating.

---

# **How does the network get updated at the end?** 
In **batch learning**, we don’t update the network after every single training example. Instead:
   - For each example, we compute its gradient.
   - At the end of the batch, we average the gradients.
   - We then update the parameters once using the averaged gradients.

This way, we get a more stable update since we’re considering multiple examples at once rather than just one. However, we need to process the entire batch before making any updates, which can slow things down compared to SGD.
