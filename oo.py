import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Callable


class Layer(ABC):
    
    def __init__(self):
        self.trainable_variables = []
    
    @abstractmethod
    def forward(self, x):
        pass
    
    def get_params(self):
        return self.trainable_variables


class DenseLayer(Layer):
    
    def __init__(self, input_dim, output_dim, activation=None, seed=42):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        
        limit = np.sqrt(6.0 / (input_dim + output_dim))
        np.random.seed(seed)
        
        self.W = tf.Variable(
            np.random.uniform(-limit, limit, (input_dim, output_dim)).astype('float32'),
            name='weight'
        )
        self.b = tf.Variable(np.zeros(output_dim, dtype='float32'), name='bias')
        
        self.trainable_variables = [self.W, self.b]
    
    def forward(self, x):
        z = tf.matmul(x, self.W) + self.b
        if self.activation is not None:
            return self.activation(z)
        return z


class DropoutLayer(Layer):
    
    def __init__(self, rate=0.5):
        super().__init__()
        self.rate = rate
        self.training = True
    
    def forward(self, x):
        if self.training and self.rate > 0:
            return tf.nn.dropout(x, rate=self.rate)
        return x
    
    def set_training(self, training):
        self.training = training


class Activation:
    
    @staticmethod
    def relu(x):
        return tf.maximum(0.0, x)
    
    @staticmethod
    def softmax(x):
        exp_x = tf.exp(x - tf.reduce_max(x, axis=1, keepdims=True))
        return exp_x / tf.reduce_sum(exp_x, axis=1, keepdims=True)


class LossFunction(ABC):
    
    @abstractmethod
    def compute(self, y_pred, y_true):
        pass


class CategoricalCrossEntropy(LossFunction):
    
    def compute(self, logits, y_true):
        y_pred = Activation.softmax(logits)
        y_pred = tf.clip_by_value(y_pred, 1e-10, 1.0)
        loss = -tf.reduce_mean(tf.reduce_sum(y_true * tf.math.log(y_pred), axis=1))
        return loss


class Metric(ABC):
    
    def __init__(self):
        self.reset()
    
    @abstractmethod
    def update(self, y_pred, y_true):
        pass
    
    @abstractmethod
    def result(self):
        pass
    
    @abstractmethod
    def reset(self):
        pass


class AccuracyMetric(Metric):
    
    def reset(self):
        self.total = 0.0
        self.count = 0
    
    def update(self, logits, y_true):
        y_pred = Activation.softmax(logits)
        predictions = tf.argmax(y_pred, axis=1)
        labels = tf.argmax(y_true, axis=1)
        correct = tf.reduce_sum(tf.cast(tf.equal(predictions, labels), tf.float32))
        
        self.total += correct.numpy()
        self.count += y_true.shape[0]
    
    def result(self):
        return self.total / self.count if self.count > 0 else 0.0


class Optimizer(ABC):
    
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
    
    @abstractmethod
    def apply_gradients(self, gradients, variables):
        pass


class SGD(Optimizer):
    
    def apply_gradients(self, gradients, variables):
        for grad, var in zip(gradients, variables):
            var.assign_sub(self.learning_rate * grad)


class Adam(Optimizer):
    
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = []
        self.v = []
        self.t = 0
    
    def apply_gradients(self, gradients, variables):
        if not self.m:
            self.m = [tf.Variable(tf.zeros_like(var)) for var in variables]
            self.v = [tf.Variable(tf.zeros_like(var)) for var in variables]
        
        self.t += 1
        
        for grad, var, m, v in zip(gradients, variables, self.m, self.v):
            m.assign(self.beta1 * m + (1 - self.beta1) * grad)
            v.assign(self.beta2 * v + (1 - self.beta2) * tf.square(grad))
            
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            
            var.assign_sub(self.learning_rate * m_hat / (tf.sqrt(v_hat) + self.epsilon))


class ClassifierModel:
    
    def __init__(self, layers: List[Layer], loss_fn: LossFunction, 
                 optimizer: Optimizer, metrics: List[Metric]):
        self.layers = layers
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.metrics = metrics
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def get_all_params(self):
        params = []
        for layer in self.layers:
            params.extend(layer.get_params())
        return params
    
    def train_step(self, batch_x, batch_y):
        with tf.GradientTape() as tape:
            logits = self.forward(batch_x)
            loss = self.loss_fn.compute(logits, batch_y)
        
        params = self.get_all_params()
        gradients = tape.gradient(loss, params)
        self.optimizer.apply_gradients(gradients, params)
        
        for metric in self.metrics:
            metric.update(logits, batch_y)
        
        return loss
    
    def evaluate(self, dataset):
        for metric in self.metrics:
            metric.reset()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch_x, batch_y in dataset:
            logits = self.forward(batch_x)
            loss = self.loss_fn.compute(logits, batch_y)
            
            total_loss += loss.numpy()
            num_batches += 1
            
            for metric in self.metrics:
                metric.update(logits, batch_y)
        
        results = {
            'loss': total_loss / num_batches,
            'metrics': {type(m).__name__: m.result() for m in self.metrics}
        }
        
        return results
    
    def set_training(self, training):
        for layer in self.layers:
            if hasattr(layer, 'set_training'):
                layer.set_training(training)


class Callback(ABC):
    
    def on_epoch_begin(self, epoch):
        pass
    
    def on_epoch_end(self, epoch, logs):
        pass
    
    def on_batch_end(self, batch, logs):
        pass


class LoggingCallback(Callback):
    
    def on_epoch_end(self, epoch, logs):
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in logs.items()])
        print(f"Epoch {epoch} - {metrics_str}")


class EarlyStoppingCallback(Callback):
    
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.wait = 0
        self.stopped = False
    
    def on_epoch_end(self, epoch, logs):
        val_loss = logs.get('val_loss', float('inf'))
        
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                self.stopped = True


class Trainer:
    
    def __init__(self, model: ClassifierModel, callbacks: List[Callback] = None):
        self.model = model
        self.callbacks = callbacks or []
    
    def train(self, train_dataset, val_dataset, epochs):
        print("\n=== OBJECT-ORIENTED TRAINING ===")
        print(f"Epochs: {epochs}\n")
        
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            
            for callback in self.callbacks:
                callback.on_epoch_begin(epoch)
            
            self.model.set_training(True)
            for metric in self.model.metrics:
                metric.reset()
            
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_x, batch_y in train_dataset:
                loss = self.model.train_step(batch_x, batch_y)
                epoch_loss += loss.numpy()
                num_batches += 1
            
            self.model.set_training(False)
            val_results = self.model.evaluate(val_dataset)
            
            logs = {
                'loss': epoch_loss / num_batches,
                'val_loss': val_results['loss'],
                'time': time.time() - epoch_start
            }
            logs.update({f"val_{k}": v for k, v in val_results['metrics'].items()})
            
            for callback in self.callbacks:
                callback.on_epoch_end(epoch, logs)
            
            if any(hasattr(cb, 'stopped') and cb.stopped for cb in self.callbacks):
                break


class ModelFactory:
    
    @staticmethod
    def create_mlp_classifier(layer_sizes, learning_rate=0.01, 
                              optimizer_type='sgd', seed=42):
        layers = []
        
        for i in range(len(layer_sizes) - 2):
            layers.append(DenseLayer(
                layer_sizes[i], 
                layer_sizes[i+1], 
                activation=Activation.relu,
                seed=seed + i
            ))
        
        layers.append(DenseLayer(
            layer_sizes[-2], 
            layer_sizes[-1], 
            activation=None,
            seed=seed + len(layer_sizes)
        ))
        
        if optimizer_type.lower() == 'adam':
            optimizer = Adam(learning_rate)
        else:
            optimizer = SGD(learning_rate)
        
        model = ClassifierModel(
            layers=layers,
            loss_fn=CategoricalCrossEntropy(),
            optimizer=optimizer,
            metrics=[AccuracyMetric()]
        )
        
        return model


class DataLoader:
    
    @staticmethod
    def load_mnist():
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        
        x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
        x_test = x_test.reshape(-1, 784).astype('float32') / 255.0
        
        y_train = tf.keras.utils.to_categorical(y_train, 10).astype('float32')
        y_test = tf.keras.utils.to_categorical(y_test, 10).astype('float32')
        
        return (x_train, y_train), (x_test, y_test)
    
    @staticmethod
    def create_dataset(x, y, batch_size=64, shuffle=True):
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.batch(batch_size)
        return dataset


if __name__ == "__main__":
    config = {
        'layer_sizes': [784, 128, 64, 10],
        'epochs': 5,
        'learning_rate': 0.01,
        'batch_size': 64,
        'optimizer': 'sgd',
        'seed': 42
    }
    
    np.random.seed(config['seed'])
    tf.random.set_seed(config['seed'])
    
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = DataLoader.load_mnist()
    
    train_dataset = DataLoader.create_dataset(x_train, y_train, config['batch_size'])
    test_dataset = DataLoader.create_dataset(x_test, y_test, config['batch_size'], shuffle=False)
    
    model = ModelFactory.create_mlp_classifier(
        layer_sizes=config['layer_sizes'],
        learning_rate=config['learning_rate'],
        optimizer_type=config['optimizer'],
        seed=config['seed']
    )
    
    callbacks = [
        LoggingCallback(),
        EarlyStoppingCallback(patience=3)
    ]
    
    trainer = Trainer(model, callbacks)
    trainer.train(train_dataset, test_dataset, config['epochs'])
    
    model.set_training(False)
    test_results = model.evaluate(test_dataset)
    print(f"\nFinal Test Results:")
    print(f"  Loss: {test_results['loss']:.4f}")
    for metric_name, value in test_results['metrics'].items():
        print(f"  {metric_name}: {value:.4f}")