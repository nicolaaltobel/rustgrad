//! A simple oxidized version of an autograd engine, packed with a Multilayer Perceptron on top.


mod tensor;
mod nn;
/*
pub use layer::Activation;
pub use multi_layer_perceptron::MLP;
*/

pub use tensor::Tensor;
pub use nn::mlp::{Mlp, Init, Activation};