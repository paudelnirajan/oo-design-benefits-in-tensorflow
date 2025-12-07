# Demonstrating OO Design Benefits in TensorFlow

A comprehensive implementation comparing **Procedural** and **Object-Oriented** programming paradigms for building deep neural networks with TensorFlow, demonstrating the practical benefits of OO design patterns in machine learning applications.

## Overview

This repository contains two complete implementations of an MNIST digit classifier:

1. **Procedural Implementation** (`procedural.py`) - Uses functions and dictionaries
2. **Object-Oriented Implementation** (`oo.py`) - Uses classes and design patterns

Both implementations achieve equivalent performance (~93% accuracy) while demonstrating the architectural and maintainability advantages of object-oriented design.

## Key Features

### Procedural Approach
- Function-based architecture
- Dictionary-based parameter management
- Linear control flow
- Direct implementation of training loop

### Object-Oriented Approach
- **Composite Pattern**: Hierarchical layer composition
- **Strategy Pattern**: Pluggable loss functions and optimizers
- **Factory Pattern**: Centralized model creation
- **Template Method**: Structured training algorithm
- **Observer Pattern**: Event-driven callbacks

## Performance Comparison

| Metric | Procedural | Object-Oriented |
|--------|-----------|-----------------|
| Test Accuracy | 93.09% | 93.19% |
| Training Time (5 epochs) | 15.11s | 15.77s |
| Lines to Add Layer | ~15 (multiple files) | ~5 (one class) |
| Testability | Function-level | Class-level |
| Extensibility | Modify multiple functions | Add new classes |

## Architecture

### Procedural Implementation Structure
```
Data Loading → Weight Initialization → Forward Pass → Loss Computation
    ↓
Gradient Computation → Parameter Update → Training Loop → Evaluation
```

### Object-Oriented Implementation Structure
```
DataLoader
    ↓
ModelFactory → ClassifierModel (Composite)
    ├── Layer (Abstract)
    │   ├── DenseLayer
    │   └── DropoutLayer
    ├── LossFunction (Strategy)
    │   └── CategoricalCrossEntropy
    ├── Optimizer (Strategy)
    │   ├── SGD
    │   └── Adam
    └── Metric
        └── AccuracyMetric
    ↓
Trainer + Callbacks (Observer)
    ├── LoggingCallback
    └── EarlyStoppingCallback
```

## Quick Start

### Installation

```bash
# Clone or download the repository
cd Paper

# Install dependencies
pip3 install -r requirements.txt
```

### Requirements
- Python 3.8+
- TensorFlow 2.10.0+
- NumPy 1.21.0+

### Running the Implementations

#### Run Procedural Implementation
```bash
python3 procedural.py
```

**Expected Output:**
```
Loading MNIST dataset...

=== PROCEDURAL TRAINING ===
Architecture: [784, 128, 64, 10]
Epochs: 5, Learning Rate: 0.01

Epoch 1/5 - Loss: 0.8678, Val Loss: 0.4116, Val Acc: 0.8858, Time: 3.12s
...
Final Test Accuracy: 0.9332
```

#### Run Object-Oriented Implementation
```bash
python3 oo.py
```

**Expected Output:**
```
Loading MNIST dataset...

=== OBJECT-ORIENTED TRAINING ===
Epochs: 5

Epoch 1 - loss: 0.8505, val_loss: 0.4049, val_AccuracyMetric: 0.8901
...
Final Test Results:
  Loss: 0.2344
  AccuracyMetric: 0.9319
```

#### Run Side-by-Side Comparison
```bash
python3 compare.py
```

This will run both implementations sequentially and provide a detailed comparison of accuracy, training time, and design benefits.

#### Run Unit Tests
```bash
python3 -m unittest test.py -v
```

**Test Coverage:**
- 11 comprehensive unit tests
- Tests for both procedural and OO implementations
- Validation of equivalent behavior

## File Structure

```
.
├── procedural.py          # Procedural implementation
├── oo.py                  # Object-oriented implementation
├── test.py                # Unit tests for both approaches
├── compare.py             # Comparison script
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Design Patterns Explained

### 1. Composite Pattern (Layers)
Allows treating individual layers and compositions of layers uniformly.

```python
class Layer(ABC):
    @abstractmethod
    def forward(self, x): pass

class DenseLayer(Layer):
    def forward(self, x):
        return activation(x @ W + b)

model.forward(x)  # Automatically calls all layers
```

### 2. Strategy Pattern (Loss & Optimizer)
Enables runtime selection of algorithms.

```python
class LossFunction(ABC):
    @abstractmethod
    def compute(self, y_pred, y_true): pass

class CategoricalCrossEntropy(LossFunction):
    def compute(self, logits, y_true):
        # Implementation
```

### 3. Factory Pattern (Model Creation)
Centralizes object creation logic.

```python
model = ModelFactory.create_mlp_classifier(
    layer_sizes=[784, 128, 64, 10],
    learning_rate=0.01,
    optimizer_type='sgd'
)
```

### 4. Template Method (Training)
Defines algorithm structure while allowing customization.

```python
class Trainer:
    def train(self, train_dataset, val_dataset, epochs):
        for epoch in range(epochs):
            self._notify_epoch_begin(epoch)
            self._train_epoch()
            self._validate()
            self._notify_epoch_end(epoch)
```

### 5. Observer Pattern (Callbacks)
Enables event-driven behavior without tight coupling.

```python
callbacks = [
    LoggingCallback(),
    EarlyStoppingCallback(patience=3)
]
trainer = Trainer(model, callbacks)
```

## Educational Use Cases

### Adding a New Layer (OO Approach)

```python
class BatchNormLayer(Layer):
    def __init__(self):
        super().__init__()
        self.gamma = tf.Variable(tf.ones(...))
        self.beta = tf.Variable(tf.zeros(...))
        self.trainable_variables = [self.gamma, self.beta]
    
    def forward(self, x):
        mean, var = tf.nn.moments(x, axes=[0])
        return self.gamma * (x - mean) / tf.sqrt(var + 1e-5) + self.beta
```

### Adding a New Optimizer (OO Approach)

```python
class RMSprop(Optimizer):
    def __init__(self, learning_rate=0.001, decay=0.9):
        super().__init__(learning_rate)
        self.decay = decay
        self.cache = []
    
    def apply_gradients(self, gradients, variables):
        if not self.cache:
            self.cache = [tf.Variable(tf.zeros_like(v)) for v in variables]
        
        for grad, var, cache in zip(gradients, variables, self.cache):
            cache.assign(self.decay * cache + (1 - self.decay) * tf.square(grad))
            var.assign_sub(self.learning_rate * grad / (tf.sqrt(cache) + 1e-8))
```

### Adding a Custom Callback (OO Approach)

```python
class ModelCheckpointCallback(Callback):
    def __init__(self, filepath):
        self.filepath = filepath
        self.best_loss = float('inf')
    
    def on_epoch_end(self, epoch, logs):
        val_loss = logs.get('val_loss')
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            # Save model weights
            print(f"Saving model at epoch {epoch}")
```

## Extending the Code

The OO implementation makes it easy to:

1. **Add new layer types** - Inherit from `Layer` base class
2. **Implement new optimizers** - Inherit from `Optimizer` base class
3. **Create custom loss functions** - Inherit from `LossFunction` base class
4. **Add training callbacks** - Inherit from `Callback` base class
5. **Build complex architectures** - Use `ModelFactory` or compose layers directly

## Testing

The test suite includes:

- **Weight initialization tests** - Verify correct shapes and values
- **Forward pass tests** - Validate output dimensions
- **Loss computation tests** - Ensure proper loss calculation
- **Training tests** - Confirm loss decreases during training
- **Metric tests** - Validate accuracy computation
- **Callback tests** - Test early stopping and logging
- **Equivalence tests** - Compare procedural and OO outputs

## Design Benefits Demonstrated

### Modularity
✓ Clear separation of concerns (data, model, training, evaluation)

### Extensibility
✓ Easy to add new components without modifying existing code

### Testability
✓ Individual classes can be unit tested in isolation

### Maintainability
✓ Changes are localized to specific classes

### Readability
✓ Self-documenting code structure with clear abstractions

### Reusability
✓ Components can be reused across different projects

## Citation

If you use this code for teaching or research, please cite:

```bibtex
@misc{paudel2025oodesign,
  author = {Paudel, Nirajan and Aruru, Gunabhiram and Kumar, Achintya},
  title = {Demonstrating OO Design Benefits in TensorFlow: 
           Applying Object-Oriented Principles to Deep Neural Network Development},
  year = {2025},
  institution = {University of Colorado Boulder}
}
```

## Authors

- **Nirajan Paudel**
- **Gunabhiram Aruru**
- **Achintya Kumar**

University of Colorado Boulder

## License

MIT License - Feel free to use for educational purposes.

## Contributing

This is an educational project. Feel free to:
- Report issues
- Suggest improvements
- Use as a template for teaching OO design patterns
- Extend with additional design patterns

## Support

For questions or discussions about this implementation, please refer to the paper or contact the authors through the University of Colorado Boulder.

---

**Note**: This implementation is designed for educational purposes to demonstrate object-oriented design principles in machine learning. For production use, consider using established frameworks like Keras or PyTorch that already implement these patterns.
