use crate::tensor::Tensor;
use ndarray::Array;
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;
use rand::distr::Uniform;

/// Wraps a vector of `Neuron` along with an `Activation`.
pub(crate) struct Layer {
    weights : Tensor,
    biases : Tensor,
}

impl Layer {
    #[allow(clippy::cast_precision_loss)]
    pub(crate) fn new_he(nin: usize, nout: usize) -> Layer {
        let std = (2.0 / nin as f64).sqrt();
        let dist = Normal::new(0., std).expect("Unable to create uniform");

        let w = Tensor::leaf(
            Array::random((nin, nout), dist).into_dyn()
        );
        let b = Tensor::leaf(
            Array::zeros((nout,)).into_dyn()
        );

        Layer {  weights : w, biases : b}
    }

    #[allow(clippy::cast_precision_loss)]
    pub(crate) fn new_xavier(nin : usize, nout : usize) -> Layer {
        let limit = (6. / (nin + nout) as f64).sqrt();
        let dist = Uniform::new(-limit, limit).expect("Unable to create uniform");

        let w = Tensor::leaf(
            Array::random((nin, nout), dist).into_dyn()
        );
        let b = Tensor::leaf(
            Array::zeros((nout,)).into_dyn()
        );

        Layer {  weights : w, biases : b}
    }

    /// Performs the forward operation of the input `x`. Returns a vector of `Value`.
    pub(crate) fn forward(&self, x: &Tensor) -> Tensor {
        x.matmul(&self.weights).add(&self.biases)
    }

    /// Returns the list of parameters of the `Layer`.
    pub(crate) fn parameters(&self) -> Vec<Tensor> {
        vec![self.weights.clone(), self.biases.clone()]
    }
}
