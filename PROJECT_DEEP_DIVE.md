# Project Deep Dive: Neural Network from Scratch in NumPy

This document explains the technical details behind the project. It is written as a deeper walkthrough of what the code is doing and why each part exists.

The project is based on MNIST digit recognition. Each image is a 28 × 28 grayscale handwritten digit. The goal is to classify each image as one of 10 digits:

```text
0, 1, 2, 3, 4, 5, 6, 7, 8, 9
```

The main model is a fully connected neural network built from scratch in NumPy.

## Table of contents

- [1. Data representation](#1-data-representation)
- [2. Normalizing pixel values](#2-normalizing-pixel-values)
- [3. Network architecture](#3-network-architecture)
- [4. Parameters: weights and biases](#4-parameters-weights-and-biases)
- [5. He initialization](#5-he-initialization)
- [6. Forward propagation](#6-forward-propagation)
  - [6.1 First layer](#61-first-layer)
  - [6.2 Output layer](#62-output-layer)
- [7. ReLU](#7-relu)
- [8. Softmax](#8-softmax)
- [9. One-hot encoding](#9-one-hot-encoding)
- [10. Cross-entropy loss](#10-cross-entropy-loss)
- [11. L2 regularization](#11-l2-regularization)
- [12. Backpropagation](#12-backpropagation)
  - [12.1 Output layer gradient](#121-output-layer-gradient)
  - [12.2 Gradients for w2 and b2](#122-gradients-for-w2-and-b2)
  - [12.3 Hidden layer gradient](#123-hidden-layer-gradient)
  - [12.4 Gradients for w1 and b1](#124-gradients-for-w1-and-b1)
- [13. Gradient descent](#13-gradient-descent)
- [14. Mini-batch gradient descent](#14-mini-batch-gradient-descent)
- [15. Momentum optimizer](#15-momentum-optimizer)
- [16. Gradient checking](#16-gradient-checking)
- [17. Model saving and loading](#17-model-saving-and-loading)
- [18. Prediction examples](#18-prediction-examples)
- [19. CNN extension](#19-cnn-extension)
  - [19.1 Convolution](#191-convolution)
  - [19.2 ReLU after convolution](#192-relu-after-convolution)
  - [19.3 Max pooling](#193-max-pooling)
  - [19.4 CNN architecture](#194-cnn-architecture)
- [20. Limitations](#20-limitations)
- [21. Future work](#21-future-work)

<!-- END_TOC -->

---

# 1. Data representation

Each MNIST image is 28 × 28 pixels.

```text
28 × 28 = 784
```

The model flattens each image into a vector of 784 pixel values.

Original image shape:

```text
28 × 28
```

Flattened vector:

```text
784 × 1
```

In the code, many images are stored together in a matrix:

```text
X shape = 784 × m
```

where `m` is the number of examples.

Each column of `X` is one image.

The labels are stored as:

```text
Y shape = m
```

For example:

```text
Y = [7, 2, 1, 0, 4, ...]
```

---

# 2. Normalizing pixel values

Raw pixel values are between 0 and 255.

```text
0   = black
255 = white
```

The code divides by 255:

```python
X = X / 255.0
```

This changes the range to:

```text
0 to 1
```

This helps training because neural networks usually learn better when inputs are on a smaller scale.

---

# 3. Network architecture

The main architecture is:

```text
784 → hidden_size → 10
```

Meaning:

```text
784 input values
hidden_size hidden neurons
10 output neurons
```

The 10 output neurons represent the digits 0 to 9.

If `hidden_size = 10`, the architecture is:

```text
784 → 10 → 10
```

If `hidden_size = 64`, the architecture becomes:

```text
784 → 64 → 10
```

---

# 4. Parameters: weights and biases

The model learns four main parameters:

```text
w1, b1, w2, b2
```

Their shapes are:

```text
w1: hidden_size × 784
b1: hidden_size × 1

w2: 10 × hidden_size
b2: 10 × 1
```

`w1` and `b1` control the hidden layer.

`w2` and `b2` control the output layer.

---

# 5. He initialization

Before training starts, weights need initial values.

Bad initialization can make learning unstable:

```text
weights too large → activations explode
weights too small → activations vanish
```

This project uses He initialization because the hidden layer uses ReLU.

For a layer with `fan_in` inputs:

```text
W = random_normal * sqrt(2 / fan_in)
```

For the first layer:

```text
fan_in = 784
```

So:

```text
w1 = random_normal * sqrt(2 / 784)
```

For the second layer:

```text
fan_in = hidden_size
```

So:

```text
w2 = random_normal * sqrt(2 / hidden_size)
```

Biases start at zero:

```text
b1 = 0
b2 = 0
```

The idea is to start the network with weights that are random but not too large or too small.

---

# 6. Forward propagation

Forward propagation is the prediction step.

The input image goes forward through the network:

```text
X → hidden layer → output layer → prediction
```

In equations:

```text
z1 = w1X + b1
a1 = ReLU(z1)

z2 = w2a1 + b2
a2 = softmax(z2)
```

## 6.1 First layer

```text
z1 = w1X + b1
```

This calculates raw hidden layer values.

Then:

```text
a1 = ReLU(z1)
```

This applies the activation function.

## 6.2 Output layer

```text
z2 = w2a1 + b2
```

This calculates raw output scores.

Then:

```text
a2 = softmax(z2)
```

This converts scores into probabilities.

---

# 7. ReLU

ReLU stands for Rectified Linear Unit.

```text
ReLU(x) = max(0, x)
```

Examples:

```text
ReLU(-3) = 0
ReLU(0)  = 0
ReLU(5)  = 5
```

ReLU gives the network non-linearity. Without a non-linear activation function, stacking layers would still behave like one big linear transformation.

The derivative is:

```text
ReLU'(x) = 1 if x > 0
ReLU'(x) = 0 if x <= 0
```

This derivative is used during backpropagation.

---

# 8. Softmax

The output layer produces 10 raw scores.

Example:

```text
z2 = [1.2, 0.5, 3.1, 0.2, ...]
```

These are not probabilities yet.

Softmax converts them into probabilities that add to 1.

For class `i`:

```text
softmax(z_i) = exp(z_i) / sum(exp(z_j))
```

The code uses a stable version:

```python
z = z - np.max(z, axis=0, keepdims=True)
```

This avoids extremely large exponentials.

After softmax, the model output might look like:

```text
[0.01, 0.02, 0.04, 0.80, 0.01, 0.02, 0.01, 0.05, 0.02, 0.02]
```

The highest probability becomes the predicted digit.

---

# 9. One-hot encoding

The true label might be:

```text
3
```

But the model output has 10 probabilities.

So the label is converted to a one-hot vector:

```text
[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
```

This makes the true label the same shape as the prediction.

For many examples:

```text
one_hot_Y shape = 10 × m
```

---

# 10. Cross-entropy loss

The loss function measures how wrong the model is.

For classification, this project uses cross-entropy loss.

For one example:

```text
loss = -log(probability assigned to the correct class)
```

If the correct digit is 3 and the model gives digit 3 probability 0.9:

```text
loss = -log(0.9)
loss ≈ 0.105
```

Small loss.

If the model gives digit 3 probability 0.01:

```text
loss = -log(0.01)
loss ≈ 4.605
```

Large loss.

For many examples, the code averages this over all examples.

---

# 11. L2 regularization

L2 regularization discourages the model from making weights too large.

The normal loss is:

```text
cross_entropy_loss
```

With L2 regularization:

```text
loss = cross_entropy_loss + L2 penalty
```

The penalty is:

```text
lambda_reg / (2m) * (sum(w1^2) + sum(w2^2))
```

This encourages smaller weights and can help reduce overfitting.

Biases are not regularized in this implementation.

During backpropagation, L2 adds this to the weight gradients:

```text
dw1 = dw1 + (lambda_reg / m) * w1
dw2 = dw2 + (lambda_reg / m) * w2
```

---

# 12. Backpropagation

Backpropagation calculates gradients.

A gradient tells us:

```text
If I slightly change this parameter, how does the loss change?
```

The goal is to calculate:

```text
dw1, db1, dw2, db2
```

These tell the optimizer how to update the parameters.

## 12.1 Output layer gradient

With softmax + cross-entropy, the output gradient simplifies to:

```text
dz2 = a2 - one_hot_Y
```

This means:

```text
prediction probabilities - true probabilities
```

If the model gave too much probability to a wrong class, that class gets a positive error.

If the model gave too little probability to the correct class, the correct class gets a negative error.

## 12.2 Gradients for w2 and b2

```text
dw2 = dz2 a1.T / m
db2 = sum(dz2) / m
```

`dw2` tells how to update the hidden-to-output weights.

`db2` tells how to update the output bias.

## 12.3 Hidden layer gradient

The output error is sent backwards through `w2`:

```text
dz1 = w2.T dz2 * ReLU'(z1)
```

The `ReLU'(z1)` part blocks gradients through hidden neurons that were inactive.

## 12.4 Gradients for w1 and b1

```text
dw1 = dz1 X.T / m
db1 = sum(dz1) / m
```

`dw1` tells how to update the input-to-hidden weights.

`db1` tells how to update the hidden bias.

---

# 13. Gradient descent

Gradient descent updates parameters in the direction that lowers the loss.

Basic update:

```text
w = w - alpha * dw
b = b - alpha * db
```

`alpha` is the learning rate.

A small learning rate means slow but safer learning.

A large learning rate means faster learning but can become unstable.

---

# 14. Mini-batch gradient descent

Instead of using the entire dataset for each update, the project uses mini-batches.

Example:

```text
batch_size = 64
```

This means the model trains on 64 examples, updates weights, then moves to the next 64 examples.

One epoch means the model has seen the full training dataset once.

Before each epoch, the code shuffles the data:

```python
indices = np.random.permutation(Y.size)
X = X[:, indices]
Y = Y[indices]
```

This keeps images and labels matched while changing the batch order.

Mini-batch training gives more frequent updates and is closer to real neural network training.

---

# 15. Momentum optimizer

Momentum improves gradient descent by remembering previous gradients.

Basic gradient descent uses only the current gradient:

```text
w = w - alpha * dw
```

Momentum uses a velocity:

```text
v = beta * v + (1 - beta) * dw
w = w - alpha * v
```

`beta` controls how much previous gradients matter.

A common value is:

```text
beta = 0.9
```

This means:

```text
90% previous velocity
10% current gradient
```

Momentum helps reduce noisy zig-zagging during optimization.

---

# 16. Gradient checking

Backpropagation is easy to implement incorrectly.

Gradient checking verifies some gradients numerically.

For a parameter `w`, numerical gradient approximation is:

```text
gradapprox = (loss(w + epsilon) - loss(w - epsilon)) / (2 * epsilon)
```

This is compared to the gradient from backpropagation.

If they are close, backprop is likely correct.

This is too slow for training, so the project only checks a few parameters on a tiny subset of data.

---

# 17. Model saving and loading

After training, the model saves:

```text
w1, b1, w2, b2
```

into:

```text
models/weights.npz
```

This means the trained model can be reused later without retraining from scratch.

---

# 18. Prediction examples

The project can save example predictions as images.

For a chosen test image, it:

1. extracts the image
2. makes a prediction
3. prints the predicted digit
4. prints the true label if available
5. saves the image to the `examples/` folder

This makes the project easier to show visually on GitHub.

---

# 19. CNN extension

The project also includes a CNN extension.

The dense network flattens the image immediately:

```text
28 × 28 → 784
```

A CNN keeps the image as a 2D structure.

This matters because digits are made of local visual patterns:

```text
edges
curves
loops
strokes
corners
```

## 19.1 Convolution

A convolutional filter is a small matrix, such as 3 × 3.

It slides over the image and checks each local patch.

At each position:

```text
output value = sum(patch * filter) + bias
```

This produces a feature map.

If the input is 28 × 28 and the filter is 3 × 3 with stride 1 and no padding:

```text
output size = 26 × 26
```

If there are 8 filters:

```text
output shape = 8 × 26 × 26
```

## 19.2 ReLU after convolution

After convolution, ReLU is applied:

```text
a_conv = ReLU(z_conv)
```

This keeps positive feature detections and blocks negative values.

## 19.3 Max pooling

Max pooling reduces the feature map size.

A 2 × 2 max pool keeps the largest value in each 2 × 2 region.

Example:

```text
[[1, 3],
 [2, 4]]
```

becomes:

```text
4
```

For a 26 × 26 feature map, 2 × 2 max pooling with stride 2 gives:

```text
13 × 13
```

## 19.4 CNN architecture

The CNN extension uses:

```text
Input image
→ convolution
→ ReLU
→ max pooling
→ flatten
→ dense layer
→ softmax
```

This is off by default because pure NumPy convolution loops are slow.

---

# 20. Limitations

This project is educational and intentionally uses NumPy rather than optimized deep learning libraries.

Limitations:

- The dense model is simple compared with modern neural networks.
- The CNN implementation uses Python loops, so it is slow on large datasets.
- The code is currently in one main script rather than split into modules.
- Accuracy is not expected to match optimized PyTorch/TensorFlow CNNs.

These limitations are acceptable because the goal is first-principles understanding.

---

# 21. Future work

Possible improvements:

- Refactor into multiple files
- Add validation split
- Add confusion matrix
- Add classification report
- Add Adam optimizer
- Add multiple hidden layers
- Add command-line options
- Add unit tests
- Vectorize convolution for faster CNN training
- Add a drawing interface to test custom digits
