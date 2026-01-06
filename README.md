# Javatorch_mini

> A minimal deep learning framework in Java inspired by PyTorch.  
> Supports basic Tensor operations, autograd, and training on MNIST.

---

## Project Overview

**Javatorch_mini** is an educational deep learning framework implemented from scratch in Java.  
It mimics some PyTorch APIs and provides the **most essential components** to understand the mechanics of deep learning:

- **Tensor**: Multi-dimensional array with automatic gradient tracking
- **Linear layer**: Fully connected layer
- **ReLU activation**
- **Softmax + CrossEntropy loss**
- **Autograd mechanism**: Dynamic computation graph and backward propagation
- **Basic dataset reading**: Raw MNIST images and labels
- **Training loop**: Single-sample SGD

> The project is designed for learning and demonstration purposes, not for production or large-scale training.

---

## Project Structure
```
Javatorch_mini/
│
├─ src/
│ ├─ Tensor.java # Core tensor class with gradient tracking
│ ├─ Op.java # Operator interface for autograd
│ ├─ Linear.java # Fully connected (dense) layer
│ ├─ ReLU.java # ReLU activation function
│ ├─ SoftmaxCrossEntropy.java # Combined softmax + cross-entropy loss
│ ├─ MNISTReader.java # MNIST binary file reader
│ └─ TrainMNIST.java # End-to-end MNIST training example
│
├─ data/
│ ├─ train-images-idx3-ubyte
│ ├─ train-labels-idx1-ubyte
│ ├─ t10k-images-idx3-ubyte
│ ├─ t10k-labels-idx1-ubyte
└─ README.md
```
