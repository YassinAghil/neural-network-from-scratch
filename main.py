import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

train_data = "train.csv"
test_data = "test.csv"
hidden_size = 10

# Main training settings
iterations = 10          # with mini-batches this now means epochs
alpha = 0.05
batch_size = 64
beta = 0.9
lambda_reg = 0.001
use_momentum = True
run_gradient_check = False

# CNN extension settings. this is off by default because pure numpy CNN training is slow.
run_cnn = False
cnn_epochs = 1
cnn_alpha = 0.01
cnn_batch_size = 16
cnn_num_filters = 8
cnn_filter_size = 3
cnn_train_examples = 500

show_images = False
save_examples = True


np.random.seed(0)


def load_data(path, shuffle=False):
    data = pd.read_csv(path).to_numpy()

    # If the file has 785 columns it is:
    # label + 784 pixels
    if data.shape[1] == 785:
        if shuffle:
            np.random.shuffle(data)

        Y = data[:, 0].astype(int)
        X = data[:, 1:]

    # If the file has 784 columns, it is:
    # 784 pixels only no labels
    elif data.shape[1] == 784:
        Y = None
        X = data

    else:
        raise ValueError(
            f"Unexpected number of columns in {path}: {data.shape[1]}. "
            "Expected 785 columns for labelled data or 784 columns for unlabelled data."
        )

    X = X / 255.0
    X = X.T

    return X, Y


def init_param(hidden_size):
    # He initialization: works better with ReLU than random values from -0.5 to 0.5
    w1 = np.random.randn(hidden_size, 784) * np.sqrt(2 / 784)
    b1 = np.zeros((hidden_size, 1))

    w2 = np.random.randn(10, hidden_size) * np.sqrt(2 / hidden_size)
    b2 = np.zeros((10, 1))

    return w1, b1, w2, b2


def relu(z):
    return np.maximum(z, 0)


def softmax(z):
    z = z - np.max(z, axis=0, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)


def d_relu(z):
    return z > 0


def forward_prop(w1, b1, w2, b2, X):
    z1 = np.dot(w1, X) + b1
    a1 = relu(z1)

    z2 = np.dot(w2, a1) + b2
    a2 = softmax(z2)

    return z1, a1, z2, a2


def one_hot(Y):
    one_hot_Y = np.zeros((10, Y.size))
    one_hot_Y[Y, np.arange(Y.size)] = 1
    return one_hot_Y


def calculate_loss(a2, Y, w1=None, w2=None, lambda_reg=0):
    if Y is None:
        raise ValueError("Loss needs labels, but Y is None.")

    m = Y.size
    one_hot_Y = one_hot(Y)

    cross_entropy = -np.sum(one_hot_Y * np.log(a2 + 1e-8)) / m
    l2_penalty = 0

    if w1 is not None and w2 is not None and lambda_reg != 0:
        l2_penalty = (lambda_reg / (2 * m)) * (np.sum(w1 ** 2) + np.sum(w2 ** 2))

    loss = cross_entropy + l2_penalty
    return loss


def back_prop(z1, a1, z2, a2, w1, w2, X, Y, lambda_reg=0):
    m = Y.size
    one_hot_Y = one_hot(Y)

    dz2 = a2 - one_hot_Y
    dw2 = np.dot(dz2, np.transpose(a1)) / m
    db2 = np.sum(dz2, axis=1, keepdims=True) / m

    dz1 = np.dot(np.transpose(w2), dz2) * d_relu(z1)
    dw1 = np.dot(dz1, np.transpose(X)) / m
    db1 = np.sum(dz1, axis=1, keepdims=True) / m

    # L2 regularization only applies to weights, not biases
    if lambda_reg != 0:
        dw1 = dw1 + (lambda_reg / m) * w1
        dw2 = dw2 + (lambda_reg / m) * w2

    return dw1, db1, dw2, db2


def update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha):
    w1 = w1 - (alpha * dw1)
    b1 = b1 - (alpha * db1)
    w2 = w2 - (alpha * dw2)
    b2 = b2 - (alpha * db2)

    return w1, b1, w2, b2


def init_momentum(w1, b1, w2, b2):
    v_dw1 = np.zeros_like(w1)
    v_db1 = np.zeros_like(b1)
    v_dw2 = np.zeros_like(w2)
    v_db2 = np.zeros_like(b2)

    return v_dw1, v_db1, v_dw2, v_db2


def update_params_momentum(w1, b1, w2, b2, dw1, db1, dw2, db2,
                           v_dw1, v_db1, v_dw2, v_db2, alpha, beta):
    v_dw1 = beta * v_dw1 + (1 - beta) * dw1
    v_db1 = beta * v_db1 + (1 - beta) * db1

    v_dw2 = beta * v_dw2 + (1 - beta) * dw2
    v_db2 = beta * v_db2 + (1 - beta) * db2

    w1 = w1 - (alpha * v_dw1)
    b1 = b1 - (alpha * v_db1)

    w2 = w2 - (alpha * v_dw2)
    b2 = b2 - (alpha * v_db2)

    return w1, b1, w2, b2, v_dw1, v_db1, v_dw2, v_db2


def get_preds(a2):
    preds = np.argmax(a2, axis=0)
    return preds


def get_accuracy(preds, Y):
    return np.sum(preds == Y) / Y.size


def plot_training(losses, accuracies):
    os.makedirs("plots", exist_ok=True)

    plt.figure()
    plt.plot(losses)
    plt.xlabel("Checkpoints")
    plt.ylabel("Loss")
    plt.title("Training loss")
    plt.savefig("plots/loss_curve.png")
    plt.close()

    plt.figure()
    plt.plot(accuracies)
    plt.xlabel("Checkpoints")
    plt.ylabel("Accuracy")
    plt.title("Training accuracy")
    plt.savefig("plots/accuracy_curve.png")
    plt.close()

    print("Saved plots to plots/loss_curve.png and plots/accuracy_curve.png")


def save_model(w1, b1, w2, b2, path="models/weights.npz"):
    os.makedirs("models", exist_ok=True)
    np.savez(path, w1=w1, b1=b1, w2=w2, b2=b2)
    print(f"Saved model to {path}")


def load_model(path="models/weights.npz"):
    data = np.load(path)
    w1 = data["w1"]
    b1 = data["b1"]
    w2 = data["w2"]
    b2 = data["b2"]
    return w1, b1, w2, b2


def gradient_check(X, Y, hidden_size, lambda_reg=0, epsilon=1e-7):
    print("Running gradient check...")

    X_small = X[:, :5]
    Y_small = Y[:5]

    w1, b1, w2, b2 = init_param(hidden_size)

    z1, a1, z2, a2 = forward_prop(w1, b1, w2, b2, X_small)
    dw1, db1, dw2, db2 = back_prop(z1, a1, z2, a2, w1, w2, X_small, Y_small, lambda_reg)

    checks = [
        ["w1", w1, dw1, (0, 0)],
        ["w1", w1, dw1, (min(1, w1.shape[0] - 1), min(10, w1.shape[1] - 1))],
        ["b1", b1, db1, (0, 0)],
        ["w2", w2, dw2, (0, 0)],
        ["w2", w2, dw2, (min(1, w2.shape[0] - 1), min(1, w2.shape[1] - 1))],
        ["b2", b2, db2, (0, 0)],
    ]

    max_difference = 0
    max_abs_difference = 0

    for name, param, grad, index in checks:
        old_value = param[index]

        param[index] = old_value + epsilon
        z1, a1, z2, a2 = forward_prop(w1, b1, w2, b2, X_small)
        loss_plus = calculate_loss(a2, Y_small, w1, w2, lambda_reg)

        param[index] = old_value - epsilon
        z1, a1, z2, a2 = forward_prop(w1, b1, w2, b2, X_small)
        loss_minus = calculate_loss(a2, Y_small, w1, w2, lambda_reg)

        param[index] = old_value

        grad_approx = (loss_plus - loss_minus) / (2 * epsilon)
        backprop_grad = grad[index]

        abs_difference = abs(backprop_grad - grad_approx)
        difference = abs_difference / (abs(backprop_grad) + abs(grad_approx) + 1e-8)
        max_difference = max(max_difference, difference)
        max_abs_difference = max(max_abs_difference, abs_difference)

        print(name, index, "backprop:", backprop_grad, "numeric:", grad_approx, "difference:", difference)

    print("Max relative gradient check difference:", max_difference)
    print("Max absolute gradient check difference:", max_abs_difference)

    if max_difference < 1e-4 or max_abs_difference < 1e-6:
        print("Gradient check passed")
    else:
        print("Gradient check warning: difference is higher than expected")


def train(X, Y, hidden_size, iterations, alpha, batch_size=64, beta=0.9,
          lambda_reg=0, use_momentum=True):
    if Y is None:
        raise ValueError("Training data must have labels.")

    w1, b1, w2, b2 = init_param(hidden_size)

    if use_momentum:
        v_dw1, v_db1, v_dw2, v_db2 = init_momentum(w1, b1, w2, b2)

    losses = []
    accuracies = []

    for i in range(iterations):
        indices = np.random.permutation(Y.size)
        X = X[:, indices]
        Y = Y[indices]

        for start in range(0, Y.size, batch_size):
            end = start + batch_size

            X_batch = X[:, start:end]
            Y_batch = Y[start:end]

            z1, a1, z2, a2 = forward_prop(w1, b1, w2, b2, X_batch)

            dw1, db1, dw2, db2 = back_prop(
                z1, a1, z2, a2, w1, w2, X_batch, Y_batch, lambda_reg
            )

            if use_momentum:
                w1, b1, w2, b2, v_dw1, v_db1, v_dw2, v_db2 = update_params_momentum(
                    w1, b1, w2, b2,
                    dw1, db1, dw2, db2,
                    v_dw1, v_db1, v_dw2, v_db2,
                    alpha, beta
                )
            else:
                w1, b1, w2, b2 = update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha)

        z1, a1, z2, a2 = forward_prop(w1, b1, w2, b2, X)
        predictions = get_preds(a2)
        accuracy = get_accuracy(predictions, Y)
        loss = calculate_loss(a2, Y, w1, w2, lambda_reg)

        losses.append(loss)
        accuracies.append(accuracy)

        print("epoch:", i, "accuracy:", accuracy, "loss:", loss)

    plot_training(losses, accuracies)
    save_model(w1, b1, w2, b2)

    return w1, b1, w2, b2


def make_preds(X, w1, b1, w2, b2):
    z1, a1, z2, a2 = forward_prop(w1, b1, w2, b2, X)
    return get_preds(a2)


def test_preds(index, X, Y, w1, b1, w2, b2):
    current_image = X[:, index, None]

    prediction = make_preds(current_image, w1, b1, w2, b2)

    print("Prediction:", prediction[0])

    if Y is not None:
        label = Y[index]
        print("True label:", label)
    else:
        print("True label: not available")

    image = current_image.reshape(28, 28)
    plt.imshow(image, cmap="gray")

    if save_examples:
        os.makedirs("examples", exist_ok=True)
        plt.savefig(f"examples/prediction_{index}.png")
        print(f"Saved example to examples/prediction_{index}.png")

    if show_images:
        plt.show()

    plt.close()


def save_predictions(predictions, path="submission.csv"):
    image_ids = np.arange(1, predictions.size + 1)

    submission = pd.DataFrame({
        "ImageId": image_ids,
        "Label": predictions
    })

    submission.to_csv(path, index=False)
    print(f"Saved predictions to {path}")


# CNN extension from scratch

def X_to_images(X):
    return X.T.reshape(X.shape[1], 28, 28)


def init_cnn_params(num_filters=8, filter_size=3):
    conv_filters = np.random.randn(num_filters, filter_size, filter_size) * np.sqrt(2 / (filter_size * filter_size))
    conv_b = np.zeros((num_filters, 1, 1))

    conv_out_size = 28 - filter_size + 1
    pool_out_size = conv_out_size // 2
    flat_size = num_filters * pool_out_size * pool_out_size

    w3 = np.random.randn(10, flat_size) * np.sqrt(2 / flat_size)
    b3 = np.zeros((10, 1))

    return conv_filters, conv_b, w3, b3


def conv_forward(X_img, conv_filters, conv_b):
    m = X_img.shape[0]
    num_filters = conv_filters.shape[0]
    filter_size = conv_filters.shape[1]
    out_size = X_img.shape[1] - filter_size + 1

    z_conv = np.zeros((m, num_filters, out_size, out_size))

    for i in range(m):
        for f in range(num_filters):
            for row in range(out_size):
                for col in range(out_size):
                    patch = X_img[i, row:row + filter_size, col:col + filter_size]
                    z_conv[i, f, row, col] = np.sum(patch * conv_filters[f]) + conv_b[f, 0, 0]

    return z_conv


def max_pool_forward(a_conv, pool_size=2, stride=2):
    m = a_conv.shape[0]
    num_filters = a_conv.shape[1]
    h = a_conv.shape[2]
    w = a_conv.shape[3]

    out_h = (h - pool_size) // stride + 1
    out_w = (w - pool_size) // stride + 1

    pool = np.zeros((m, num_filters, out_h, out_w))
    pool_mask = np.zeros_like(a_conv)

    for i in range(m):
        for f in range(num_filters):
            for row in range(out_h):
                for col in range(out_w):
                    row_start = row * stride
                    col_start = col * stride
                    patch = a_conv[i, f, row_start:row_start + pool_size, col_start:col_start + pool_size]

                    max_value = np.max(patch)
                    pool[i, f, row, col] = max_value

                    max_index = np.argmax(patch)
                    max_row, max_col = np.unravel_index(max_index, patch.shape)
                    pool_mask[i, f, row_start + max_row, col_start + max_col] = 1

    return pool, pool_mask


def cnn_forward(X, conv_filters, conv_b, w3, b3):
    X_img = X_to_images(X)

    z_conv = conv_forward(X_img, conv_filters, conv_b)
    a_conv = relu(z_conv)

    pool, pool_mask = max_pool_forward(a_conv)
    flat = pool.reshape(pool.shape[0], -1).T

    z3 = np.dot(w3, flat) + b3
    a3 = softmax(z3)

    return X_img, z_conv, a_conv, pool, pool_mask, flat, z3, a3


def max_pool_back(dpool, pool_mask, pool_size=2, stride=2):
    da_conv = np.zeros_like(pool_mask)

    m = dpool.shape[0]
    num_filters = dpool.shape[1]
    out_h = dpool.shape[2]
    out_w = dpool.shape[3]

    for i in range(m):
        for f in range(num_filters):
            for row in range(out_h):
                for col in range(out_w):
                    row_start = row * stride
                    col_start = col * stride
                    mask_patch = pool_mask[i, f, row_start:row_start + pool_size, col_start:col_start + pool_size]
                    da_conv[i, f, row_start:row_start + pool_size, col_start:col_start + pool_size] += mask_patch * dpool[i, f, row, col]

    return da_conv


def conv_back(X_img, dz_conv, conv_filters, lambda_reg=0):
    m = X_img.shape[0]
    num_filters = conv_filters.shape[0]
    filter_size = conv_filters.shape[1]
    out_size = dz_conv.shape[2]

    dfilters = np.zeros_like(conv_filters)
    dconv_b = np.zeros((num_filters, 1, 1))

    for i in range(m):
        for f in range(num_filters):
            for row in range(out_size):
                for col in range(out_size):
                    patch = X_img[i, row:row + filter_size, col:col + filter_size]
                    dfilters[f] += dz_conv[i, f, row, col] * patch
                    dconv_b[f, 0, 0] += dz_conv[i, f, row, col]

    dfilters = dfilters / m
    dconv_b = dconv_b / m

    if lambda_reg != 0:
        dfilters = dfilters + (lambda_reg / m) * conv_filters

    return dfilters, dconv_b


def cnn_back_prop(X_img, z_conv, a_conv, pool, pool_mask, flat, z3, a3,
                  conv_filters, w3, Y, lambda_reg=0):
    m = Y.size
    one_hot_Y = one_hot(Y)

    dz3 = a3 - one_hot_Y
    dw3 = np.dot(dz3, flat.T) / m
    db3 = np.sum(dz3, axis=1, keepdims=True) / m

    if lambda_reg != 0:
        dw3 = dw3 + (lambda_reg / m) * w3

    dflat = np.dot(w3.T, dz3)
    dpool = dflat.T.reshape(pool.shape)

    da_conv = max_pool_back(dpool, pool_mask)
    dz_conv = da_conv * d_relu(z_conv)

    dfilters, dconv_b = conv_back(X_img, dz_conv, conv_filters, lambda_reg)

    return dfilters, dconv_b, dw3, db3


def calculate_cnn_loss(a3, Y, conv_filters=None, w3=None, lambda_reg=0):
    m = Y.size
    one_hot_Y = one_hot(Y)
    cross_entropy = -np.sum(one_hot_Y * np.log(a3 + 1e-8)) / m
    l2_penalty = 0

    if conv_filters is not None and w3 is not None and lambda_reg != 0:
        l2_penalty = (lambda_reg / (2 * m)) * (np.sum(conv_filters ** 2) + np.sum(w3 ** 2))

    return cross_entropy + l2_penalty


def update_cnn_params(conv_filters, conv_b, w3, b3, dfilters, dconv_b, dw3, db3, alpha):
    conv_filters = conv_filters - alpha * dfilters
    conv_b = conv_b - alpha * dconv_b
    w3 = w3 - alpha * dw3
    b3 = b3 - alpha * db3

    return conv_filters, conv_b, w3, b3


def train_cnn(X, Y, epochs, alpha, batch_size=16, num_filters=8, filter_size=3, lambda_reg=0):
    if Y is None:
        raise ValueError("CNN training data must have labels.")

    conv_filters, conv_b, w3, b3 = init_cnn_params(num_filters, filter_size)

    for epoch in range(epochs):
        indices = np.random.permutation(Y.size)
        X = X[:, indices]
        Y = Y[indices]

        for start in range(0, Y.size, batch_size):
            end = start + batch_size

            X_batch = X[:, start:end]
            Y_batch = Y[start:end]

            X_img, z_conv, a_conv, pool, pool_mask, flat, z3, a3 = cnn_forward(X_batch, conv_filters, conv_b, w3, b3)

            dfilters, dconv_b, dw3, db3 = cnn_back_prop(
                X_img, z_conv, a_conv, pool, pool_mask, flat, z3, a3,
                conv_filters, w3, Y_batch, lambda_reg
            )

            conv_filters, conv_b, w3, b3 = update_cnn_params(
                conv_filters, conv_b, w3, b3,
                dfilters, dconv_b, dw3, db3,
                alpha
            )

        X_img, z_conv, a_conv, pool, pool_mask, flat, z3, a3 = cnn_forward(X, conv_filters, conv_b, w3, b3)
        preds = get_preds(a3)
        accuracy = get_accuracy(preds, Y)
        loss = calculate_cnn_loss(a3, Y, conv_filters, w3, lambda_reg)

        print("cnn epoch:", epoch, "accuracy:", accuracy, "loss:", loss)

    return conv_filters, conv_b, w3, b3


def make_cnn_preds(X, conv_filters, conv_b, w3, b3):
    X_img, z_conv, a_conv, pool, pool_mask, flat, z3, a3 = cnn_forward(X, conv_filters, conv_b, w3, b3)
    return get_preds(a3)


if __name__ == "__main__":
    X, Y = load_data(train_data, shuffle=True)

    print("Training X shape:", X.shape)
    print("Training Y shape:", Y.shape)

    if run_gradient_check:
        gradient_check(X, Y, hidden_size, lambda_reg)

    w1, b1, w2, b2 = train(
        X, Y,
        hidden_size,
        iterations,
        alpha,
        batch_size=batch_size,
        beta=beta,
        lambda_reg=lambda_reg,
        use_momentum=use_momentum
    )

    X_test, Y_test = load_data(test_data, shuffle=False)

    print("Test X shape:", X_test.shape)

    if Y_test is not None:
        print("Test Y shape:", Y_test.shape)
    else:
        print("Test labels: not available")

    test_predictions = make_preds(X_test, w1, b1, w2, b2)

    print("First 20 test predictions:")
    print(test_predictions[:20])

    # If test data has labels, calculate accuracy
    if Y_test is not None:
        test_accuracy = get_accuracy(test_predictions, Y_test)
        print("Test accuracy:", test_accuracy)

        test_preds(0, X_test, Y_test, w1, b1, w2, b2)
        test_preds(1, X_test, Y_test, w1, b1, w2, b2)
        test_preds(2, X_test, Y_test, w1, b1, w2, b2)

    # If test data has no labels, still show predictions and save submission
    else:
        save_predictions(test_predictions)

        test_preds(0, X_test, None, w1, b1, w2, b2)
        test_preds(1, X_test, None, w1, b1, w2, b2)
        test_preds(2, X_test, None, w1, b1, w2, b2)

    if run_cnn:
        print("Running CNN demo on a small subset because pure NumPy CNN training is slow.")
        X_cnn = X[:, :cnn_train_examples]
        Y_cnn = Y[:cnn_train_examples]

        conv_filters, conv_b, w3, b3 = train_cnn(
            X_cnn, Y_cnn,
            cnn_epochs,
            cnn_alpha,
            batch_size=cnn_batch_size,
            num_filters=cnn_num_filters,
            filter_size=cnn_filter_size,
            lambda_reg=lambda_reg
        )

        cnn_predictions = make_cnn_preds(X_cnn[:, :20], conv_filters, conv_b, w3, b3)
        print("First 20 CNN predictions on CNN training subset:")
        print(cnn_predictions)
