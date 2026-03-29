use crate::layer::{Activation, Layer};
use crate::value::Value;
use rand::rngs::StdRng;
use rand::SeedableRng;

/// The actual Multi Layer Perceptron. Contains a vector of layers.
pub struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    /// Initializes a `MLP`. The parameters are:
    /// - `nin`: the size of the inputs
    /// - `nouts`: the sizes of the hidden layers and of the output
    /// - `activations`: the activations to use for each layer
    /// # Panics
    /// If the input size is less than 1, if `nouts` is empty, or if the number of layers is different from the size of `activations`
    #[must_use]
    pub fn new(nin: usize, nouts: &[usize], activations: &[Activation]) -> MLP {
        Self::build(nin, nouts, activations, |w, act| Layer::new(w[0], w[1], *act))
    }

    /// Initializes a `MLP` using a fixed seed. The parameters are:
    /// - `nin`: the size of the inputs
    /// - `nouts`: the sizes of the hidden layers and of the output
    /// - `activations`: the activations to use for each layer
    /// # Panics
    /// If the input size is less than 1, if `nouts` is empty, or if the number of layers is different from the size of `activations`
    #[must_use]
    pub fn new_seeded(nin: usize, nouts: &[usize], activations: &[Activation], seed: u64) -> MLP {
        let mut rng = StdRng::seed_from_u64(seed);
        Self::build(nin, nouts, activations, |w, act| {
            Layer::new_with_rng(w[0], w[1], *act, &mut rng)
        })
    }

    fn build<F>(
        nin: usize,
        nouts: &[usize],
        activations: &[Activation],
        mut layer_factory: F,
    ) -> MLP
    where
        F: FnMut(&[usize], &Activation) -> Layer,
    {
        assert!(nin > 0, "MLP input size must be > 0");
        assert!(!nouts.is_empty(), "MLP must contain at least one layer");
        assert_eq!(
            nouts.len(),
            activations.len(),
            "Number of layers is different from number of activations"
        );

        let sizes: Vec<usize> = std::iter::once(nin).chain(nouts.iter().copied()).collect();
        MLP {
            layers: sizes
                .windows(2)
                .zip(activations.iter())
                .map(|(w, act)| layer_factory(w, act))
                .collect(),
        }
    }

    /// Performs the forward operation on all the layers.
    #[must_use]
    pub fn forward(&self, x: &[f64]) -> Vec<Value> {
        let mut out: Vec<Value> = Value::from(x);
        for layer in &self.layers {
            out = layer.forward(&out);
        }
        out
    }

    /// Returns all the parameters of the `MLP`.
    #[must_use]
    pub fn parameters(&self) -> Vec<Value> {
        self.layers.iter().flat_map(Layer::parameters).collect()
    }

    /// Sets the gradient of every parameter to 0.0
    pub fn zero_grad(&self) {
        self.parameters().iter().for_each(|p| p.set_grad(0.0));
    }

    /// Calls the `update` function for every parameter.
    pub fn update(&self, lr: f64) {
        self.parameters().iter().for_each(|p| p.update(lr));
    }
}

#[cfg(test)]
mod tests {
    use super::MLP;
    use crate::Activation::{ReLU, TanH};

    #[test]
    fn seeded_init_is_deterministic() {
        let a = MLP::new_seeded(2, &[4, 1], &[ReLU, TanH], 123);
        let b = MLP::new_seeded(2, &[4, 1], &[ReLU, TanH], 123);

        let ap: Vec<f64> = a.parameters().iter().map(super::super::value::Value::data).collect();
        let bp: Vec<f64> = b.parameters().iter().map(super::super::value::Value::data).collect();

        assert_eq!(ap.len(), bp.len());
        assert!(ap.iter().zip(bp.iter()).all(|(x, y)| (x - y).abs() < 0.00001));
    }

    #[test]
    fn different_seeds_produce_different_params() {
        let a = MLP::new_seeded(2, &[4, 1], &[ReLU, TanH], 123);
        let b = MLP::new_seeded(2, &[4, 1], &[ReLU, TanH], 124);

        let ap: Vec<f64> = a.parameters().iter().map(super::super::value::Value::data).collect();
        let bp: Vec<f64> = b.parameters().iter().map(super::super::value::Value::data).collect();

        assert_eq!(ap.len(), bp.len());
        assert!(ap.iter().zip(bp.iter()).any(|(x, y)| (x - y).abs() > 0.00001));
    }
}
