use crate::layer::Activation;
use crate::value::Value;
use rand::distr::{Distribution, Uniform};
use rand::Rng;

/// A struct that represents a `Neuron` of the network. Contains:
/// - `w`: the weights, represented as a vector of `Value`
/// - `b`: the bias, of type `Value`
pub(crate) struct Neuron {
    w: Vec<Value>,
    b: Value,
}

impl Neuron {
    pub(crate) fn new(nin: usize, activation: Activation) -> Self {
        let mut rng = rand::rng();
        Self::new_with_rng(nin, activation, &mut rng)
    }
    
    /// Creates a `Neuron` using a fixed seed. Used for reproducibility.
    pub(crate) fn new_with_rng<R: Rng + ?Sized>(
        nin: usize,
        activation: Activation,
        rng: &mut R,
    ) -> Self {
        match activation {
            Activation::ReLU => Self::he_init(nin, rng),
            Activation::TanH => Self::xavier_init(nin, rng),
        }
    }
    
    /// Initializes weights and bias based on the He init.
    #[allow(clippy::cast_precision_loss)]
    fn he_init<R: Rng + ?Sized>(nin: usize, rng: &mut R) -> Self {
        let scale = (2.0 / nin as f64).sqrt();
        let uniform = Uniform::new(-scale, scale).expect("invalid He init range");

        let mut w = Vec::new();

        for _i in 0..nin {
            w.push(Value::leaf(uniform.sample(rng)));
        }

        let b = Value::leaf(uniform.sample(rng));

        Neuron { w, b }
    }
    
    /// Initializes weights and bias based on the Xavier init. 
    #[allow(clippy::cast_precision_loss)]
    fn xavier_init<R: Rng + ?Sized>(nin: usize, rng: &mut R) -> Self {
        let scale = (1.0 / nin as f64).sqrt();
        let uniform = Uniform::new(-scale, scale).expect("invalid Xavier init range");

        let mut w = Vec::new();

        for _i in 0..nin {
            w.push(Value::leaf(uniform.sample(rng)));
        }

        let b = Value::leaf(uniform.sample(rng));

        Neuron { w, b }
    }

    /// Performs the forward operation on the input `x` and returns the activated `Value`.
    pub(crate) fn forward(&self, x: &[Value], activation: Activation) -> Value {
        assert_eq!(
            self.w.len(),
            x.len(),
            "Neuron input dimension mismatch: expected {}, got {}",
            self.w.len(),
            x.len()
        );

        let act = self
            .w
            .iter()
            .zip(x.iter())
            .fold(self.b.clone(), |acc, (wi, bi)| acc.add(wi.mul(bi)));

        match activation {
            Activation::ReLU => act.relu(),
            Activation::TanH => act.tanh(),
        }
    }
    
    /// Returns the list of parameters of the `Neuron`.
    pub(crate) fn parameters(&self) -> Vec<Value> {
        self.w
            .iter()
            .cloned()
            .chain(std::iter::once(self.b.clone()))
            .collect()
    }
}
