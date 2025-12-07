import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

import procedural as proc
import oo


def prepare_data(batch_size=64):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 784).astype('float32') / 255.0
    
    y_train = tf.keras.utils.to_categorical(y_train, 10).astype('float32')
    y_test = tf.keras.utils.to_categorical(y_test, 10).astype('float32')
    
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(10000).batch(batch_size)
    
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.batch(batch_size)
    
    return train_dataset, test_dataset


def run_procedural(train_dataset, test_dataset, config):
    print("\n" + "="*60)
    print("RUNNING PROCEDURAL IMPLEMENTATION")
    print("="*60)
    
    start_time = time.time()
    
    params = proc.train_procedural(config, train_dataset, test_dataset)
    
    test_loss, test_acc = proc.evaluate(params, test_dataset, config)
    
    total_time = time.time() - start_time
    
    results = {
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'total_time': total_time,
        'params': params
    }
    
    print(f"\nProcedural Results:")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  Total Time: {total_time:.2f}s")
    
    return results


def run_oo(train_dataset, test_dataset, config):
    print("\n" + "="*60)
    print("RUNNING OBJECT-ORIENTED IMPLEMENTATION")
    print("="*60)
    
    start_time = time.time()
    
    model = oo.ModelFactory.create_mlp_classifier(
        layer_sizes=config['layer_sizes'],
        learning_rate=config['learning_rate'],
        optimizer_type='sgd',
        seed=config['seed']
    )
    
    callbacks = [
        oo.LoggingCallback()
    ]
    
    trainer = oo.Trainer(model, callbacks)
    trainer.train(train_dataset, test_dataset, config['epochs'])
    
    model.set_training(False)
    test_results = model.evaluate(test_dataset)
    
    total_time = time.time() - start_time
    
    results = {
        'test_loss': test_results['loss'],
        'test_accuracy': test_results['metrics']['AccuracyMetric'],
        'total_time': total_time,
        'model': model
    }
    
    print(f"\nOO Results:")
    print(f"  Test Loss: {results['test_loss']:.4f}")
    print(f"  Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"  Total Time: {total_time:.2f}s")
    
    return results


def compare_results(proc_results, oo_results):
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    print(f"\nAccuracy:")
    print(f"  Procedural: {proc_results['test_accuracy']:.4f}")
    print(f"  OO:         {oo_results['test_accuracy']:.4f}")
    print(f"  Difference: {abs(proc_results['test_accuracy'] - oo_results['test_accuracy']):.4f}")
    
    print(f"\nTraining Time:")
    print(f"  Procedural: {proc_results['total_time']:.2f}s")
    print(f"  OO:         {oo_results['total_time']:.2f}s")
    print(f"  Difference: {abs(proc_results['total_time'] - oo_results['total_time']):.2f}s")
    
    print(f"\nDesign Benefits of OO Approach:")
    print(f"  ✓ Clear separation of concerns (Layer, Model, Optimizer)")
    print(f"  ✓ Easy to extend (add new layer types, optimizers)")
    print(f"  ✓ Testable units (each class can be tested independently)")
    print(f"  ✓ Callbacks for flexible training customization")
    print(f"  ✓ Design patterns: Composite, Strategy, Factory, Observer")


def main():
    config = {
        'layer_sizes': [784, 128, 64, 10],
        'num_layers': 3,
        'epochs': 5,
        'learning_rate': 0.01,
        'batch_size': 64,
        'seed': 42
    }
    
    np.random.seed(config['seed'])
    tf.random.set_seed(config['seed'])
    
    print("Preparing MNIST dataset...")
    train_dataset, test_dataset = prepare_data(config['batch_size'])
    
    proc_results = run_procedural(train_dataset, test_dataset, config)
    
    train_dataset, test_dataset = prepare_data(config['batch_size'])
    oo_results = run_oo(train_dataset, test_dataset, config)
    
    compare_results(proc_results, oo_results)


if __name__ == "__main__":
    main()