import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import time


def load_and_preprocess_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 784).astype('float32') / 255.0
    
    y_train = tf.keras.utils.to_categorical(y_train, 10).astype('float32')
    y_test = tf.keras.utils.to_categorical(y_test, 10).astype('float32')
    
    return (x_train, y_train), (x_test, y_test)


def create_batches(x, y, batch_size=64, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    return dataset


def initialize_weights(layer_sizes, seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    params = {}
    for i in range(len(layer_sizes) - 1):
        limit = np.sqrt(6.0 / (layer_sizes[i] + layer_sizes[i+1]))
        params[f'W{i+1}'] = tf.Variable(
            np.random.uniform(-limit, limit, 
                            (layer_sizes[i], layer_sizes[i+1])).astype('float32')
        )
        params[f'b{i+1}'] = tf.Variable(np.zeros(layer_sizes[i+1], dtype='float32'))
    
    return params


def relu(x):
    return tf.maximum(0.0, x)


def softmax(x):
    exp_x = tf.exp(x - tf.reduce_max(x, axis=1, keepdims=True))
    return exp_x / tf.reduce_sum(exp_x, axis=1, keepdims=True)


def forward(params, x, num_layers):
    a = x
    for i in range(1, num_layers):
        z = tf.matmul(a, params[f'W{i}']) + params[f'b{i}']
        a = relu(z)
    
    logits = tf.matmul(a, params[f'W{num_layers}']) + params[f'b{num_layers}']
    return logits


def cross_entropy_loss(logits, y_true):
    y_pred = softmax(logits)
    y_pred = tf.clip_by_value(y_pred, 1e-10, 1.0)
    loss = -tf.reduce_mean(tf.reduce_sum(y_true * tf.math.log(y_pred), axis=1))
    return loss


def compute_accuracy(logits, y_true):
    y_pred = softmax(logits)
    predictions = tf.argmax(y_pred, axis=1)
    labels = tf.argmax(y_true, axis=1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))
    return accuracy


def compute_gradients(params, x, y, num_layers):
    with tf.GradientTape() as tape:
        logits = forward(params, x, num_layers)
        loss = cross_entropy_loss(logits, y)
    
    gradients = tape.gradient(loss, list(params.values()))
    return loss, gradients


def apply_gradients(params, gradients, learning_rate):
    for param, grad in zip(params.values(), gradients):
        param.assign_sub(learning_rate * grad)


def train_one_batch(params, batch_x, batch_y, config):
    loss, gradients = compute_gradients(params, batch_x, batch_y, config['num_layers'])
    apply_gradients(params, gradients, config['learning_rate'])
    return loss


def evaluate(params, dataset, config):
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0
    
    for batch_x, batch_y in dataset:
        logits = forward(params, batch_x, config['num_layers'])
        loss = cross_entropy_loss(logits, batch_y)
        acc = compute_accuracy(logits, batch_y)
        
        total_loss += loss.numpy()
        total_acc += acc.numpy()
        num_batches += 1
    
    return total_loss / num_batches, total_acc / num_batches


def train_procedural(config, train_dataset, val_dataset):
    print("\n=== PROCEDURAL TRAINING ===")
    print(f"Architecture: {config['layer_sizes']}")
    print(f"Epochs: {config['epochs']}, Learning Rate: {config['learning_rate']}\n")
    
    params = initialize_weights(config['layer_sizes'], seed=config['seed'])
    
    for epoch in range(config['epochs']):
        epoch_start = time.time()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_x, batch_y in train_dataset:
            loss = train_one_batch(params, batch_x, batch_y, config)
            epoch_loss += loss.numpy()
            num_batches += 1
        
        val_loss, val_acc = evaluate(params, val_dataset, config)
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch+1}/{config['epochs']} - "
              f"Loss: {epoch_loss/num_batches:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Acc: {val_acc:.4f}, "
              f"Time: {epoch_time:.2f}s")
    
    return params


if __name__ == "__main__":
    config = {
        'layer_sizes': [784, 128, 64, 10],
        'num_layers': 3,
        'epochs': 5,
        'learning_rate': 0.01,
        'batch_size': 64,
        'seed': 42
    }
    
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    
    train_dataset = create_batches(x_train, y_train, config['batch_size'])
    test_dataset = create_batches(x_test, y_test, config['batch_size'], shuffle=False)
    
    trained_params = train_procedural(config, train_dataset, test_dataset)
    
    test_loss, test_acc = evaluate(trained_params, test_dataset, config)
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")