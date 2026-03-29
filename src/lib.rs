//! A simple oxidized version of an autograd engine, packed with a Multilayer Perceptron on top.


mod layer;
mod multi_layer_perceptron;
mod neuron;
mod value;

pub use layer::Activation;
pub use multi_layer_perceptron::MLP;
pub use value::Value;
