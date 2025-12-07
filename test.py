import unittest
import numpy as np
import tensorflow as tf

import procedural
import oo


class TestProceduralImplementation(unittest.TestCase):
    
    def setUp(self):
        self.config = {
            'layer_sizes': [784, 128, 64, 10],
            'num_layers': 3,
            'learning_rate': 0.01,
            'seed': 42
        }
    
    def test_weight_initialization(self):
        params = procedural.initialize_weights(self.config['layer_sizes'], seed=42)
        
        self.assertIn('W1', params)
        self.assertIn('b1', params)
        self.assertIn('W2', params)
        self.assertIn('b2', params)
        self.assertIn('W3', params)
        self.assertIn('b3', params)
        
        self.assertEqual(params['W1'].shape, (784, 128))
        self.assertEqual(params['b1'].shape, (128,))
        self.assertEqual(params['W3'].shape, (64, 10))
    
    def test_forward_pass_shape(self):
        params = procedural.initialize_weights(self.config['layer_sizes'])
        x = tf.random.normal((32, 784))
        
        logits = procedural.forward(params, x, self.config['num_layers'])
        
        self.assertEqual(logits.shape, (32, 10))
    
    def test_loss_computation(self):
        logits = tf.random.normal((32, 10))
        y_true = tf.one_hot(tf.random.uniform((32,), 0, 10, dtype=tf.int32), 10)
        
        loss = procedural.cross_entropy_loss(logits, y_true)
        
        self.assertGreater(loss.numpy(), 0)
        self.assertEqual(loss.shape, ())
    
    def test_single_batch_training(self):
        params = procedural.initialize_weights(self.config['layer_sizes'])
        x = tf.random.normal((64, 784))
        y = tf.one_hot(tf.random.uniform((64,), 0, 10, dtype=tf.int32), 10)
        
        logits = procedural.forward(params, x, self.config['num_layers'])
        loss_before = procedural.cross_entropy_loss(logits, y).numpy()
        
        for _ in range(10):
            loss, grads = procedural.compute_gradients(params, x, y, self.config['num_layers'])
            procedural.apply_gradients(params, grads, 0.1)
        
        logits = procedural.forward(params, x, self.config['num_layers'])
        loss_after = procedural.cross_entropy_loss(logits, y).numpy()
        
        self.assertLess(loss_after, loss_before)


class TestOOImplementation(unittest.TestCase):
    
    def setUp(self):
        self.layer_sizes = [784, 128, 64, 10]
    
    def test_dense_layer_creation(self):
        layer = oo.DenseLayer(784, 128, activation=oo.Activation.relu, seed=42)
        
        self.assertEqual(layer.W.shape, (784, 128))
        self.assertEqual(layer.b.shape, (128,))
        self.assertEqual(len(layer.trainable_variables), 2)
    
    def test_dense_layer_forward(self):
        layer = oo.DenseLayer(784, 128, activation=oo.Activation.relu)
        x = tf.random.normal((32, 784))
        
        output = layer.forward(x)
        
        self.assertEqual(output.shape, (32, 128))
        self.assertTrue(tf.reduce_all(output >= 0))
    
    def test_model_creation(self):
        model = oo.ModelFactory.create_mlp_classifier(
            self.layer_sizes, 
            learning_rate=0.01,
            seed=42
        )
        
        x = tf.random.normal((32, 784))
        logits = model.forward(x)
        
        self.assertEqual(logits.shape, (32, 10))
    
    def test_train_step(self):
        model = oo.ModelFactory.create_mlp_classifier(
            self.layer_sizes, 
            learning_rate=0.1,
            seed=42
        )
        
        x = tf.random.normal((64, 784))
        y = tf.one_hot(tf.random.uniform((64,), 0, 10, dtype=tf.int32), 10)
        
        logits = model.forward(x)
        loss_before = model.loss_fn.compute(logits, y).numpy()
        
        for _ in range(10):
            loss = model.train_step(x, y)
        
        logits = model.forward(x)
        loss_after = model.loss_fn.compute(logits, y).numpy()
        
        self.assertLess(loss_after, loss_before)
    
    def test_accuracy_metric(self):
        metric = oo.AccuracyMetric()
        
        logits = tf.constant([[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]])
        y_true = tf.constant([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
        
        metric.update(logits, y_true)
        accuracy = metric.result()
        
        self.assertAlmostEqual(accuracy, 1.0, places=5)
    
    def test_callbacks(self):
        early_stop = oo.EarlyStoppingCallback(patience=2)
        
        early_stop.on_epoch_end(1, {'val_loss': 0.5})
        self.assertFalse(early_stop.stopped)
        
        early_stop.on_epoch_end(2, {'val_loss': 0.6})
        early_stop.on_epoch_end(3, {'val_loss': 0.7})
        
        self.assertTrue(early_stop.stopped)


class TestComparison(unittest.TestCase):
    
    def test_equivalent_outputs(self):
        layer_sizes = [784, 64, 10]
        seed = 42
        
        proc_params = procedural.initialize_weights(layer_sizes, seed=seed)
        
        oo_model = oo.ModelFactory.create_mlp_classifier(layer_sizes, seed=seed)
        
        x = tf.random.normal((32, 784), seed=seed)
        
        proc_logits = procedural.forward(proc_params, x, num_layers=2)
        oo_logits = oo_model.forward(x)
        
        self.assertEqual(proc_logits.shape, oo_logits.shape)


if __name__ == '__main__':
    unittest.main()