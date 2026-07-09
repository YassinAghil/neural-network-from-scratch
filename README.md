# Neural Network from Scratch in NumPy for MNIST Digit Recognition

A from-scratch implementation of a neural network for handwritten digit recognition using **NumPy only**.

This project started as a simple MNIST classifier with one hidden layer inspired by [Samson Zhang](https://youtu.be/w8yWXqWQYmU?si=eEkvhAFVrxVrAv4Z) then grew into a deeper learning project where I implemented the core parts of neural network training manually: forward propagation, backpropagation, softmax, cross-entropy loss, mini-batch gradient descent, momentum, L2 regularization, gradient checking, model saving/loading, training plots, and an educational CNN extension.

The goal of this project was not to get the highest possible MNIST accuracy using a deep learning library but to understand what libraries like PyTorch and TensorFlow are doing under the hood.

---

## Project summary

The main model is a fully connected neural network trained on the MNIST / Kaggle Digit Recognizer dataset.

```text
Input image: 28 × 28 pixels
Flattened input: 784 values
Hidden layer: hidden_size neurons
Hidden activation: ReLU
Output layer: 10 neurons
Output activation: Softmax
Loss: Cross-entropy + optional L2 regularization
Optimizer: Mini-batch gradient descent with optional momentum
```

The network predicts one of 10 digit classes:

```text
0, 1, 2, 3, 4, 5, 6, 7, 8, 9
```

---

## What I implemented from scratch

- Data loading for labelled and unlabelled CSV files
- Pixel normalization
- He initialization for ReLU networks
- ReLU activation and derivative
- Numerically stable softmax
- One-hot label encoding
- Forward propagation
- Cross-entropy loss
- L2 regularization
- Backpropagation
- Mini-batch gradient descent
- Momentum optimizer
- Accuracy calculation
- Loss and accuracy plotting
- Model saving and loading with `.npz` files
- Gradient checking for backpropagation verification
- Prediction examples saved as images
- Kaggle-style submission file generation
- Educational CNN extension using convolution and max pooling

---

## Dataset

This project expects CSV files in the same format as the Kaggle Digit Recognizer dataset.

### Labelled data

A labelled CSV has 785 columns:

```text
label, pixel0, pixel1, pixel2, ..., pixel783
```

### Unlabelled data

An unlabelled CSV has 784 columns:

```text
pixel0, pixel1, pixel2, ..., pixel783
```

The code automatically detects whether the file has labels or not.

The dataset files are not included in this repository because they are large. [Download](https://www.kaggle.com/c/digit-recognizer) the data separately and place the CSV files in the project root

---

## How to run

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python main.py
```

The script will:

1. Load `train.csv`
2. Train the neural network
3. Save training plots
4. Save model weights
5. Load `test.csv`
6. Make predictions
7. Calculate test accuracy if labels are available
8. Save prediction examples
9. Save `submission.csv` if the test data is unlabelled

---

## Main settings

The main training settings are near the top of `main.py`:

```python
hidden_size = 10
iterations = 10          # with mini-batches this means epochs
alpha = 0.05
batch_size = 64
beta = 0.9
lambda_reg = 0.001
use_momentum = True
run_gradient_check = True
```

| Setting | Meaning |
|---|---|
| `hidden_size` | Number of neurons in the hidden layer |
| `iterations` | Number of training epochs |
| `alpha` | Learning rate |
| `batch_size` | Number of examples per mini-batch |
| `beta` | Momentum strength |
| `lambda_reg` | L2 regularization strength |
| `use_momentum` | Turns momentum optimizer on/off |
| `run_gradient_check` | Verifies some backprop gradients numerically |

---

## Neural network architecture

The main fully connected network is:

```text
784 inputs → hidden_size hidden neurons → 10 outputs
```

Forward propagation:

```text
z1 = w1X + b1
a1 = ReLU(z1)

z2 = w2a1 + b2
a2 = softmax(z2)
```

Where:

```text
X  = input image data
w1 = weights from input to hidden layer
b1 = hidden bias
z1 = raw hidden values
a1 = hidden activations
w2 = weights from hidden to output layer
b2 = output bias
z2 = raw output scores
a2 = final probabilities
```

---

## Loss function

The model uses cross-entropy loss:

```text
loss = -mean(log(probability assigned to the correct class))
```

With L2 regularization enabled, the loss becomes:

```text
regularized_loss = cross_entropy_loss + weight_penalty
```

This encourages the model to learn useful weights without making them unnecessarily large.

---

## Backpropagation

Backpropagation calculates how much each parameter contributed to the error.

For softmax + cross-entropy, the output gradient simplifies to:

```text
dz2 = a2 - one_hot_Y
```

Then the model calculates:

```text
dw2, db2
dw1, db1
```

These gradients are used to update the weights and biases.

---

## Mini-batch gradient descent

Instead of updating weights after seeing the full dataset, the code trains on small batches:

```text
batch 1 → forward → backprop → update
batch 2 → forward → backprop → update
batch 3 → forward → backprop → update
...
```

This gives the model more frequent updates and makes training closer to how real neural networks are trained.

---

## Momentum optimizer

Momentum improves gradient descent by storing a running average of previous gradients.

Instead of updating using only the current gradient:

```text
w = w - alpha * dw
```

momentum uses:

```text
v = beta * v + (1 - beta) * dw
w = w - alpha * v
```

This helps reduce noisy zig-zagging and makes optimization smoother.

---

## He initialization

The model uses He initialization:

```text
W = random_normal * sqrt(2 / fan_in)
```

This is useful for ReLU networks because it helps keep activations and gradients at a healthy scale during training.

---

## Gradient checking

Gradient checking verifies that backpropagation is working correctly.

The idea is to compare:

```text
analytical gradient from backpropagation
```

against:

```text
numerical gradient from slightly changing a parameter and measuring the loss change
```

Numerical approximation:

```text
gradapprox = (loss(w + epsilon) - loss(w - epsilon)) / (2 * epsilon)
```

This is too slow for training but useful for checking that the backpropagation formulas are correct.

---

## CNN extension

The project also includes an educational CNN extension implemented from scratch using NumPy.

The CNN uses:

```text
image → convolution → ReLU → max pooling → flatten → dense layer → softmax
```

This is off by default because pure NumPy convolution loops are slow compared with optimized deep learning libraries.

To enable it:

```python
run_cnn = True
```

The CNN is included to show how convolution and max pooling work internally, not to compete with optimized PyTorch/TensorFlow implementations.

---

## Outputs

After running the script, the project can generate:

```text
plots/loss_curve.png
plots/accuracy_curve.png
models/weights.npz
examples/prediction_0.png
examples/prediction_1.png
examples/prediction_2.png
submission.csv
```

---

## Example project structure

```text
mnist-neural-network-from-scratch/
├── main.py
├── README.md
├── PROJECT_DEEP_DIVE.md
├── requirements.txt
├── .gitignore
├── plots/
│   ├── loss_curve.png
│   └── accuracy_curve.png
├── examples/
│   ├── prediction_0.png
│   ├── prediction_1.png
│   └── prediction_2.png
└── models/
    └── weights.npz
```

---

## Future improvements


- Split the code into separate files such as `data.py`, `model.py`, `train.py`, and `cnn.py`
- Add command-line arguments with `argparse`
- Add a validation split
- Add confusion matrix and classification report
- Add unit tests for shape checks, softmax, one-hot encoding, and forward propagation
- Add Adam optimizer from scratch
- Add support for multiple hidden layers
- Vectorize the CNN implementation for speed
- Add a small GUI or drawing canvas to test custom handwritten digits

---
