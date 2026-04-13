# rustgrad

A scalar-valued autograd engine implemented in Rust, inspired by Karpathy's [micrograd](https://github.com/karpathy/micrograd). Implements reverse-mode automatic differentiation over a dynamically built DAG, with a neural network library on top.

## Features

- Scalar-valued autograd engine with dynamic DAG construction
- Reverse-mode backpropagation via topological sort
- Neural network abstractions: `Neuron`, `Layer`, `MLP`
- Configurable activation functions per layer (`TanH`, `ReLU`, `Linear`)
- Xavier and He initialization, automatically selected depending on the chosen activation

## Architecture

Each `Value` wraps an `Rc<RefCell<ValueInner>>`, allowing shared ownership across the DAG while maintaining interior mutability for gradient accumulation. Backward closures are stored as `Rc<dyn Fn()>` and invoked in reverse topological order during backpropagation.
```
Value
└── Rc<RefCell<ValueInner>>
    ├── data: f64
    ├── grad: f64
    ├── backward: Rc<dyn Fn()>
    ├── prev: Vec<Value>
    └── op: Operation
```

## Moon Dataset

Trained a `2→32→32→32→1` MLP with `tanh` activations on a binary classification task using the moon dataset (4000 points, 80/20 train/test split).

- **Loss**: SVM hinge loss
- **Train accuracy**: ~97%
- **Test accuracy**: ~97%

![Moon dataset decision boundary](src/bin/plots/moons.png)

## Usage
### Basic example
```rust
use ferrograd::{Value, MLP, Activation};

let mlp = MLP::new(2, &[16, 16, 1], &[Activation::TanH, Activation::TanH, Activation::TanH]);

let x = vec![Value::leaf(1.0), Value::leaf(-2.0)];
let out = mlp.forward(&x);
out[0].backward();
```

### Training
```rust
// forward pass
let ypred: Vec<Value> = xs.iter().flat_map(|x| mlp.forward(x)).collect();

// hinge loss
let loss = ypred.iter().zip(ys.iter())
    .map(|(yout, ygt)| Value::leaf(1.0).sub(&yout.mul_scalar(*ygt)).relu())
    .reduce(|acc, v| acc.add(&v))
    .unwrap()
    .mul_scalar(1.0 / ys.len() as f64);

// backward pass
mlp.zero_grad();
loss.backward();
mlp.update(lr);
```

## Project Structure
```
rustgrad/
├── src/
│   ├── lib.rs          # library root
│   ├── value.rs        # Value type and autograd engine
│   ├── nn.rs           # Neuron, Layer, MLP
│   └── bin/
│       ├── basic.rs    # basic usage example
│       └── moon.rs     # moon dataset training
├── plots/
│   ├── plot_moons.py   # plotting script
│   └── moons.png       # decision boundary
└── Cargo.toml
```
