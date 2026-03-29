use crate::neuron::Neuron;
use crate::value::Value;
use rand::Rng;

#[derive(Clone, Copy)]
pub enum Activation {
    ReLU,
    TanH,
}


/// Wraps a vector of `Neuron` along with an `Activation`.
pub(crate) struct Layer {
    neurons: Vec<Neuron>,
    activation: Activation,
}

impl Layer {
    pub(crate) fn new(nin: usize, nout: usize, activation: Activation) -> Layer {
        Layer {
            neurons: (0..nout).map(|_| Neuron::new(nin, activation)).collect(),
            activation,
        }
    }

    /// Creates a `Neuron` using a fixed seed. Used for reproducibility.
    pub(crate) fn new_with_rng<R: Rng + ?Sized>(
        nin: usize,
        nout: usize,
        activation: Activation,
        rng: &mut R,
    ) -> Layer {
        Layer {
            neurons: (0..nout)
                .map(|_| Neuron::new_with_rng(nin, activation, rng))
                .collect(),
            activation,
        }
    }

    /// Performs the forward operation of the input `x`. Returns a vector of `Value`.
    pub(crate) fn forward(&self, x: &[Value]) -> Vec<Value> {
        self.neurons
            .iter()
            .map(|n| n.forward(x, self.activation))
            .collect()
    }

    /// Returns the list of parameters of the `Layer`.
    pub(crate) fn parameters(&self) -> Vec<Value> {
        self.neurons.iter().flat_map(Neuron::parameters).collect()
    }
}
