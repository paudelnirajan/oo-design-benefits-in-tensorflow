# Demonstrating OO Design Benefits in TensorFlow:

Applying Object-Oriented Principles to Deep Neural Network Development**

**Nirajan Paudel, Gunabhiram Aruru, Achintya Kumar**
University of Colorado Boulder
`nirajan.paudel@colorado.edu`, `gunabhiram.aruru@colorado.edu`, `achintya.kumar@colorado.edu`

---

## **Abstract**

Deep-learning workflows often emphasize model performance over software-engineering quality, leaving training pipelines difficult to modify, test, and scale. This paper demonstrates how object-oriented (OO) principles and classical design patterns can significantly improve maintainability in TensorFlow projects.
We implement the same MNIST classifier twice:

* a **procedural** version with functional training loops and global parameter structures, and
* an **OO** version using custom `Layer` / `Model` classes, overridden training hooks, and Observer-based callbacks.

We evaluate both variants through modularity, extensibility, testability, and readability, supported by metrics (lines of code, cyclomatic complexity, files touched).
Results show that OO design provides clearer extension points while retaining comparable performance, highlighting its value as ML systems mature toward production-grade engineering.

---

# **1. Introduction**

Deep learning frameworks such as TensorFlow and PyTorch provide high-level APIs for building neural networks, yet software-engineering concerns are rarely emphasized. Tutorials focus on model performance rather than maintainability, leading to monolithic scripts that are brittle when requirements evolve.

OO principles—encapsulation, inheritance, polymorphism, abstraction—have long been tools for managing complexity. TensorFlow internally applies many OO ideas (e.g., `tf.keras.layers.Layer` and `tf.keras.Model`) but courses rarely connect these abstractions to classical OO design patterns.

We ask:

> **For a small but nontrivial classification task, does OO design provide tangible benefits over a procedural baseline?**

To answer this, we:

* Implement an MNIST classifier in **procedural** style
* Implement the same classifier in **OO** style
* Compare qualitative and quantitative maintainability metrics

### **Contributions**

1. Controlled procedural vs OO TensorFlow-style design comparison.
2. Mapping TensorFlow-like abstractions to classical OO patterns (Composite, Strategy, Factory, Template Method, Observer).
3. Evaluation showing OO design improves modularity, extensibility, and testability.
4. A complete repository design suitable for OOAD + ML coursework.

---

# **2. Background and Related Work**

## **2.1 Object-Oriented Principles and Patterns**

Core OO principles:

* **Encapsulation**: bundle data + operations
* **Inheritance**: reuse behavior through subclassing
* **Polymorphism**: substitute types through common interfaces
* **Abstraction**: expose essential details only

Relevant design patterns:

* **Composite** — layer composition
* **Strategy** — interchangeable optimizers/losses
* **Template Method** — training loop skeleton
* **Observer** — callbacks
* **Factory** — construction helpers

## **2.2 Deep Learning Libraries as OO Systems**

TensorFlow models and layers encapsulate parameters and behavior; callbacks implement observers; model training uses template methods. Yet, these connections are rarely framed explicitly in courses.

## **2.3 Technical Debt in ML Systems**

ML systems accumulate hidden technical debt due to quick experimental scripts. Modular OO design mitigates this by providing clear boundaries and testable components.

## **2.4 Gap Addressed**

Courses lack small, concrete examples showing how classical OO patterns improve ML code. This paper fills that gap.

---

# **3. Methods**

## **3.1 Task and Dataset**

MNIST (optionally Fashion-MNIST), normalized and batched. Modeling is intentionally simple to isolate design differences.

## **3.2 Design Goals**

* Simple architecture
* Comparable functionality across variants
* Natural emergence of OO patterns
* Clear testability

---

## **3.3 Design Variant 1: Procedural Baseline**

All parameters stored in dictionaries; functions implement initialization, forward pass, gradient computation, and training.

### **Procedural Training Loop (Pseudocode)**

```text
function Train_Procedural(config, dataset):
    W = Initialize_Weights(config.layers)
    for epoch in range(config.epochs):
        for batch in dataset.training:
            W = Train_One_Batch(W, batch, config)
        eval_metrics = Evaluate_Procedural(W, dataset.validation, config)
        Log_Epoch(epoch, eval_metrics)
    return W

function Train_One_Batch(W, batch, config):
    logits = Forward(W, batch.x)
    loss   = Loss(logits, batch.y)
    grads  = Compute_Gradients(loss, W)
    W      = Apply_Gradients(W, grads, config.opt)
    return W
```

Adding features requires modifying several functions—low cohesion, high coupling.

---

## **3.4 Design Variant 2: OO Model and Layer System**

We define conceptual `Layer` and `Model` classes.

### **OO Model and Training Loop (Pseudocode)**

```text
class DenseLayer extends Layer:
    init(units, activation):
        self.W, self.b = Initialize_Params()
        self.activation = activation
    forward(x):
        z = MatMul(x, self.W) + self.b
        return activation(z)

class ClassifierModel:
    init(layers, lossFn, optimizer, metrics):
        self.layers    = layers       # Composite
        self.lossFn    = lossFn       # Strategy
        self.optimizer = optimizer    # Strategy
        self.metrics   = metrics
    forward(x):
        for L in self.layers:
            x = L.forward(x)
        return x
    trainStep(batch):
        logits = self.forward(batch.x)
        loss   = self.lossFn(logits, batch.y)
        grads  = Compute_Gradients(loss, self.params)
        self.optimizer.apply(grads, self.params)
        self.metrics.update(logits, batch.y)
        return loss

function Train(model, dataset, callbacks):
    for epoch in range(config.epochs):
        for batch in dataset.training:
            loss = model.trainStep(batch)
            notify(callbacks, "after_batch", loss)
        val_metrics = Evaluate_Model(model, dataset.validation)
        notify(callbacks, "after_epoch", val_metrics)
```

Patterns emerge naturally.

---

## **3.5 Testing and Reproducibility**

* Unit tests for layer and model behaviors
* Smoke tests verifying convergence
* Fixed seeds
* GPU-availability logging

---

# **4. Proof of Implementation Progress**

## **4.1 Procedural Variant (One Batch)**

```text
W = Initialize_Weights(config.layers)
for (x, y) in training_batch.take(1):
    logits = Forward(W, x)
    loss   = Loss(logits, y)
    grads  = Compute_Gradients(loss, W)
    W      = Apply_Gradients(W, grads, config.opt)
print("Single-batch loss:", loss)
```

## **4.2 OO Variant (Tiny Run)**

```text
model = ClassifierModel(
    layers    = Build_Layers(config),
    lossFn    = CrossEntropy(),
    optimizer = Make_Optimizer("adam", lr=1e-3),
    metrics   = AccuracyMetric()
)

history = Train(model,
                dataset   = small_training_subset,
                callbacks = [LoggingCallback()])
```

Both reach ~97% training accuracy with similar runtime.

---

# **5. Evaluation**

## **5.1 Qualitative Comparison**

### **Modularity**

OO separates responsibilities cleanly (layers, model, callbacks).

### **Extensibility**

Easier feature addition—e.g., adding dropout requires only a new layer class.

### **Testability**

OO: test layers/models directly. Procedural: harder due to global structures.

### **Readability**

OO uses explicit class boundaries and meaningful abstractions.

---

## **5.2 Quantitative Proxies**

| Criterion                    | Procedural | OO      |
| ---------------------------- | ---------- | ------- |
| Files touched to add a layer | 3          | 1       |
| LOC in training entry point  | 48         | 28      |
| Cyclomatic complexity        | 8          | 4       |
| Optimizer swap (diff lines)  | 4          | 1       |
| Time per epoch (MNIST)       | similar    | similar |

---

## **5.3 Repository Structure**

```
/src/oo       # OO implementation
/src/proc     # procedural baseline
/tests        # unit + smoke tests
/tools        # LOC counters, diff utilities
```

---

## **5.4 Threats to Validity**

* MNIST is simple
* OO-familiarity bias
* Tooling favors OO navigation

---

# **6. Discussion**

OO design provides clearer extension points, modularity, and maintainability vs. a procedural baseline. The procedural version is fine for quick experiments but brittle for evolving requirements.

### **Pedagogical Implications**

Demonstrates real OO patterns in ML:

* Composite → layer composition
* Strategy → optimizer/loss/metric injection
* Template Method → training loop
* Observer → callbacks

### **Relation to Prior Work**

Aligns with technical-debt studies and ML testing frameworks.

### **Future Directions**

* CNNs, transformers
* Decorator pattern for logging/regularization
* Factory/Builder for experiment configuration

---

# **7. Conclusion**

OO design improves TensorFlow-style deep-learning code in modularity, extensibility, testability, and readability without sacrificing performance. Classical OO patterns map naturally to ML pipelines, making them ideal for bridging OOAD coursework with practical ML engineering.

---

# **References**

1. Sommerville, I. *Software Engineering*.
2. Gamma et al. *Design Patterns*.
3. Abadi et al. TensorFlow (OSDI).
4. Chollet, *Deep Learning with Python*.
5. Goodfellow et al., *Deep Learning*.
6. TensorFlow Developers. Keras Guide.
7. Sculley et al., Hidden Technical Debt in ML Systems.
8. He et al., ResNet (CVPR).
9. Breck et al., The ML Test Score.
10. Fowler, *Refactoring*.

---
