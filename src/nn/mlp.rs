use crate::nn::layer::Layer;
use crate::tensor::{IntoTensor, Tensor};

/// The actual Multi Layer Perceptron. Contains a vector of layers.
pub struct Mlp {
    layers: Vec<Layer>,
    activation: Activation
}

#[derive(Copy, Clone)]
pub enum Init{
    Xavier,
    He
}

#[derive(Copy, Clone)]
pub enum Activation{
    ReLU,
    Tanh
}

impl Mlp {
    /// Initializes a `MLP`. The parameters are:
    /// - `nin`: the size of the inputs
    /// - `nouts`: the sizes of the hidden layers and of the output
    /// - `activations`: the activations to use for each layer
    /// # Panics
    /// If the input size is less than 1, if `nouts` is empty, or if the number of layers is different from the size of `activations`
    #[must_use]
    pub fn new(layer_sizes: &[usize], init : Init, activation : Activation) -> Mlp {
        let layers = layer_sizes
            .windows(2)
            .map(|w| match init {
                Init::Xavier => Layer::new_xavier(w[0], w[1]),
                Init::He => Layer::new_he(w[0], w[1]),
            })
            .collect();
        Mlp {layers, activation}
    }

    /// Performs the forward operation on all the layers.
    /// # Panics
    /// TODO
    #[must_use]
    pub fn forward(&self, x: impl IntoTensor) -> Tensor {
        let x = x.into_tensor();

        let x = if x.data().ndim() == 1 {
            let len = x.data().len();
            Tensor::leaf(x.data().into_shape_with_order((1,len)).unwrap().into_dyn())
        } else {
            x
        };

        self.layers
            .iter()
            .enumerate()
            .fold(x, |acc, (i, layer)| {
                let out = layer.forward(&acc);
                if i < self.layers.len() -1 {
                    match self.activation {
                        Activation::ReLU => out.relu(),
                        Activation::Tanh => out.tanh(),
                    }
                } else {
                    out
                }
            })
    }

    /// Returns all the parameters of the `MLP`.
    #[must_use]
    pub fn parameters(&self) -> Vec<Tensor> {
        self.layers.iter().flat_map(Layer::parameters).collect()
    }

    /// Sets the gradient of every parameter to 0.0
    pub fn zero_grad(&self) {
        self.parameters().iter().for_each(Tensor::zero_grad);
    }

    /// Calls the `update` function for every parameter.
    pub fn update(&self, lr: f64) {
        self.parameters().iter().for_each(|p| p.update(lr));
    }
}