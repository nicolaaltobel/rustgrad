use ndarray::{Array, Axis, IxDyn};
use std::cell::RefCell;
use std::collections::HashSet;
use std::fmt::{Debug, Display, Formatter};
use std::rc::Rc;

pub trait IntoTensor {
    fn into_tensor(self) -> Tensor;
}

/// The inner fields of a `Tensor`.
struct TensorInner {
        data: Array<f64, IxDyn>,
        grad: Array<f64, IxDyn>,
        backward: Rc<dyn Fn()>,
        prev: Vec<Tensor>,
}

/// The type used for operations inside the model.
/// It allows to keep track of the operations and children that generated the value,
/// which is needed for backpropagation.
#[derive(Clone)]
pub struct Tensor {
    inner: Rc<RefCell<TensorInner>>,
}

impl Tensor {
    /// Creates a `Tensor` given its data and vector of children.
    pub(crate) fn new(data: Array<f64, IxDyn>, children: Vec<Tensor>) -> Tensor {
        Tensor {
            inner: Rc::new(RefCell::new(TensorInner {
                grad: Array::zeros(data.raw_dim()),
                data,
                prev: children,
                backward: Rc::new(|| {}),
            })),
        }
    }

    /// Creates a `Tensor` with no children, given its data.
    #[must_use]
    pub fn leaf(data: Array<f64, IxDyn>) -> Tensor {
        Tensor {
            inner: Rc::new(RefCell::new(TensorInner {
                grad: Array::zeros(data.raw_dim()),
                data,
                prev: Vec::new(),
                backward: Rc::new(|| {}),
            })),
        }
    }
    
    /// Maps an array of `f64` into an array of `Tensor` with no children.
    #[must_use]
    pub fn from(x: Vec<f64>) -> Tensor {
        Tensor::leaf(Array::from(x).into_dyn())
    }
    
    /// Hyperbolic tangent function for type `Value`.
    /// # Panics
    /// TODO
    #[must_use]
    pub fn tanh(&self) -> Tensor {
        let t = self.inner.borrow().data.tanh();
        let out = Tensor::new(t.clone(), vec![self.clone()]);

        let out_weak = Rc::downgrade(&out.inner);
        let self_clone = self.clone();

        let backward = {
            move || {
                let grad_out = out_weak
                    .upgrade()
                    .expect("tanh backward graph node dropped")
                    .borrow()
                    .grad
                    .clone();

                let local_grad = t.mapv(|v| 1. - v.powi(2));
                self_clone.inner.borrow_mut().grad += &(local_grad * &grad_out);
            }
        };

        out.inner.borrow_mut().backward = Rc::new(backward);

        out
    }
    
    /// Performs the rectified Linear Unit on the `data` of the `Tensor`.
    /// # Panics
    ///  If the upgrade of the `Weak` pointer to an `Rc` fails.
    #[must_use] 
    pub fn relu(&self) -> Tensor {
        let x = self.inner.borrow().data.clone();
        let out = Tensor::new(x.mapv(|v| v.max(0.0)), vec![self.clone()]);

        let out_weak = Rc::downgrade(&out.inner);
        let self_clone = self.clone();

        let backward = {
            move || {
                let grad_out = out_weak
                    .upgrade()
                    .expect("Failed to upgrade Rc")
                    .borrow()
                    .grad
                    .clone();

                let local_grad = x.mapv(|v| if v > 0. {1.} else {0.});
                self_clone.inner.borrow_mut().grad += &(local_grad * &grad_out);
            }
        };
        out.inner.borrow_mut().backward = Rc::new(backward);

        out
    }

    /// Performs a topological ordering of the sequence of `Tensor` that generated `Self`
    /// and then recursively calls itself on the ordered vector, performing backpropagation of the gradient.
    pub fn backward(&self) {
        let mut topo: Vec<Tensor> = Vec::new();
        let mut visited: HashSet<*const RefCell<TensorInner>> = HashSet::new();
        let mut stack: Vec<(Tensor, bool)> = vec![(self.clone(), false)];

        while let Some((node, expanded)) = stack.pop() {
            let ptr = Rc::as_ptr(&node.inner);

            if expanded {
                topo.push(node);
                continue;
            }

            if visited.insert(ptr) {
                stack.push((node.clone(), true));

                let prev = node.inner.borrow().prev.clone();
                for child in prev.into_iter().rev() {
                    stack.push((child, false));
                }
            }
        }
        let shape = self.inner.borrow().data.shape().to_vec();
        self.inner.borrow_mut().grad = Array::ones(shape).into_dyn();

        for v in topo.iter().rev() {
            let backward = v.inner.borrow().backward.clone();
            backward();
        }
    }
    
    /// Performs addition of `Self` with a type that implements `IntoTensor`, which include `Tensor`, `&Tensor`, `f64`.
    /// # Panics
    /// If the upgrade of the `Weak` pointer to an `Rc` fails.
    #[must_use]
    pub fn add(&self, other: impl IntoTensor) -> Tensor {
        let other = other.into_tensor();
        let out = Tensor::new(
            self.inner.borrow().data.clone() + other.inner.borrow().data.clone(),
            vec![self.clone(), other.clone()],
        );

        let out_weak = Rc::downgrade(&out.inner);
        let self_clone = self.clone();
        let other_clone = other.clone();
        let self_shape = self.inner.borrow().data.shape().to_vec();
        let other_shape = other.inner.borrow().data.shape().to_vec();

        let backward = {
            move || {
                let grad_out = out_weak.upgrade().unwrap().borrow().grad.clone();
                self_clone.inner.borrow_mut().grad += &sum_to_shape(&grad_out, &self_shape);
                other_clone.inner.borrow_mut().grad += &sum_to_shape(&grad_out, &other_shape);
            }
        };
        out.inner.borrow_mut().backward = Rc::new(backward);

        out
    }

    /// Performs subtraction of `Self` with a type that implements `IntoTensor`, which include `Tensor`, `&Tensor`, `f64`.
    /// # Panics
    /// If the upgrade of the `Weak` pointer to an `Rc` fails.
    #[must_use]
    pub fn sub(&self, other: impl IntoTensor) -> Tensor {
        let other = other.into_tensor();
        let out = Tensor::new(
            self.inner.borrow().data.clone() - &other.inner.borrow().data,
            vec![self.clone(), other.clone()],
        );

        let out_weak = Rc::downgrade(&out.inner);
        let self_clone = self.clone();
        let other_clone = other.clone();
        let self_shape = self.inner.borrow().data.shape().to_vec();
        let other_shape = other.inner.borrow().data.shape().to_vec();

        out.inner.borrow_mut().backward = Rc::new(move || {
            let out_grad = out_weak.upgrade().unwrap().borrow().grad.clone();
            self_clone.inner.borrow_mut().grad += &sum_to_shape(&out_grad, &self_shape);
            other_clone.inner.borrow_mut().grad -= &sum_to_shape(&out_grad, &other_shape);
        });

        out
    }
    
    /// Performs multiplication of `Self` with a type that implements `IntoTensor`, which include `Tensor`, `&Tensor`, `f64`.
    /// # Panics
    /// If the upgrade of the `Weak` pointer to an `Rc` fails.
    #[must_use]
    pub fn mul(&self, other: impl IntoTensor) -> Tensor {
        let other = other.into_tensor();
        let out = Tensor::new(
            self.inner.borrow().data.clone() * &other.inner.borrow().data,
            vec![self.clone(), other.clone()],
        );

        let out_weak = Rc::downgrade(&out.inner);
        let self_clone = self.clone();
        let other_clone = other.clone();
        let self_shape = self.inner.borrow().data.shape().to_vec();
        let other_shape = other.inner.borrow().data.shape().to_vec();

        let backward = {
            move || {
                let out_grad = out_weak.upgrade().unwrap().borrow().grad.clone();
                let grad_self = out_grad.clone() * &other_clone.inner.borrow().data;
                let grad_other = out_grad * &self_clone.inner.borrow().data;
                self_clone.inner.borrow_mut().grad += &sum_to_shape(&grad_self, &self_shape);
                other_clone.inner.borrow_mut().grad += &sum_to_shape(&grad_other,&other_shape);
            }
        };
        out.inner.borrow_mut().backward = Rc::new(backward);

        out
    }

    /// TODO
    /// # Panics
    /// TODO
    #[must_use]
    pub fn matmul(&self, other: impl IntoTensor) -> Tensor {
        let other = other.into_tensor();
        let a = self.inner.borrow().data.clone();
        let b = other.inner.borrow().data.clone();

        // Shape check
        assert_eq!(a.ndim(), 2, "Wrong number of dimensions of self");
        assert_eq!(b.ndim(), 2, "Wrong number of dimensions of other");
        assert_eq!(a.shape()[1], b.shape()[0], "Incompatible shapes");

        let a2 = a.view().into_dimensionality::<ndarray::Ix2>().unwrap();
        let b2 = b.view().into_dimensionality::<ndarray::Ix2>().unwrap();
        let out_data = a2.dot(&b2).into_dyn();

        let out = Tensor::new(out_data, vec![self.clone(), other.clone()]);

        let out_weak = Rc::downgrade(&out.inner);
        let self_clone = self.clone();
        let other_clone = other.clone();

        let backward = {
            move || {
                let grad_out = out_weak.upgrade().unwrap().borrow().grad.clone();
                let grad_out2 = grad_out.into_dimensionality::<ndarray::Ix2>().unwrap();

                let b_data = other_clone.inner.borrow().data.clone();
                let a_data = self_clone.inner.borrow().data.clone();

                let b2 = b_data.view().into_dimensionality::<ndarray::Ix2>().unwrap();
                let a2 = a_data.view().into_dimensionality::<ndarray::Ix2>().unwrap();

                // grad_a = grad_out @ b.t
                self_clone.inner.borrow_mut().grad += &grad_out2.dot(&b2.t()).into_dyn();
                // grad_b = a.t @ grad_out
                other_clone.inner.borrow_mut().grad += &a2.t().dot(&grad_out2).into_dyn();
            }
        };

        out.inner.borrow_mut().backward = Rc::new(backward);

        out
    }
    
    /// Performs the power of `Self` with an exponent whose type implements `IntoTensor`, which include `Tensor`, `&Tensor`, `f64`.
    /// # Panics
    /// If the upgrade of the `Weak` pointer to an `Rc` fails.
    #[must_use]
    pub fn pow(&self, exp: f64) -> Tensor {
        let x = self.inner.borrow().data.clone();
        let out = Tensor::new(x.powf(exp), vec![self.clone()]);

        let out_weak = Rc::downgrade(&out.inner);
        let self_clone = self.clone();

        out.inner.borrow_mut().backward = Rc::new(move || {
            let out_grad = out_weak.upgrade().unwrap().borrow().grad.clone();
            self_clone.inner.borrow_mut().grad += &(exp * x.powf(exp - 1.0) * out_grad);
        });

        out
    }

    /// TODO
    /// # Panics
    /// TODO
    #[must_use]
    pub fn log_softmax(&self, axis : usize) -> Tensor {
        assert_eq!(self.inner.borrow().data.ndim(), 2, "expected 2D input");    

        let x = self.inner.borrow().data.clone();

        let max = x.map_axis(Axis(axis), |lane| {
            lane.fold(f64::NEG_INFINITY, |a, &b| a.max(b))
        });
        let shifted = &x - &max.insert_axis(Axis(axis));

        let log_sum_exp = shifted
            .mapv(f64::exp)
            .sum_axis(Axis(axis))
            .mapv(f64::ln);
        let out_data = &shifted - &log_sum_exp.insert_axis(Axis(axis));

        let out = Tensor::new(out_data.clone(), vec![self.clone()]);
        let out_weak = Rc::downgrade(&out.inner);
        let self_clone = self.clone();

        out.inner.borrow_mut().backward = Rc::new(move || {
            let grad_out = out_weak.upgrade().unwrap().borrow().grad.clone();

            let softmax = out_data.mapv(f64::exp);
            let sum_grad = grad_out
                .sum_axis(Axis(axis))
                .insert_axis(Axis(axis));
            self_clone.inner.borrow_mut().grad += &(grad_out - softmax * sum_grad);
        });

        out
    }

    /// TODO
    /// # Panics
    /// TODO
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn nll_loss(&self, targets : &[usize]) -> Tensor{
        assert_eq!(self.inner.borrow().data.ndim(), 2, "expected 2D input");

        let x = self.inner.borrow().data.clone();
        let n = targets.len() as f64;

        let loss_val = targets
            .iter()
            .enumerate()
            .map(|(i, &t)| x[[i,t]])
            .sum::<f64>()
            / -n;

        let out = Tensor::new(
            Array::from_elem((), loss_val).into_dyn(),
            vec![self.clone()],
        );

        let out_weak = Rc::downgrade(&out.inner);
        let self_clone = self.clone();
        let targets = targets.to_vec();

        out.inner.borrow_mut().backward = Rc::new(move || {
            let grad_out = out_weak.upgrade().unwrap().borrow().grad[[]];
            let grad_scalar = grad_out / -(targets.len() as f64);
            let mut grad = Array::zeros(self_clone.inner.borrow().data.raw_dim());
            for (i, &t) in targets.iter().enumerate() {
                grad[[i,t]] += grad_scalar;
            }
            self_clone.inner.borrow_mut().grad += &grad;
        });

        out
    }

    /// Sets the gradient to `x`
    pub fn set_grad(&self, x: Array<f64, IxDyn>) {
        self.inner.borrow_mut().grad = x;
    }

    pub fn zero_grad(&self) {
        let mut inner = self.inner.borrow_mut();
        let dim = inner.data.raw_dim();
        inner.grad = Array::zeros(dim);
    }

    /// Updates `data` by `- lr * grad`
    pub fn update(&self, lr: f64) {
        let mut inner = self.inner.borrow_mut();
        let update = &(lr * &inner.grad.clone());
        inner.data -= update;
    }

    /// Returns the value of `data`
    #[must_use] 
    pub fn data(&self) -> Array<f64, IxDyn> {
        self.inner.borrow().data.clone()
    }

    /// Returns the value of `grad`
    #[must_use]
    pub fn grad(&self) -> Array<f64, IxDyn> {
        self.inner.borrow().grad.clone()
    }
}

impl IntoTensor for Tensor {
    fn into_tensor(self) -> Tensor {
        self
    }
}

impl IntoTensor for &Tensor {
    fn into_tensor(self) -> Tensor {
        self.clone()
    }
}

impl IntoTensor for Vec<f64> {
    fn into_tensor(self) -> Tensor {
        Tensor::leaf(Array::from(self).into_dyn())
    }
}

impl IntoTensor for &Vec<f64> {
    fn into_tensor(self) -> Tensor {
        Tensor::leaf(Array::from(self.clone()).into_dyn())
    }
}

impl Display for Tensor {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Tensor({})", self.inner.borrow().data)
    }
}

impl Debug for Tensor {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Tensor({})", self.inner.borrow().data)
    }
}

fn sum_to_shape(grad : &Array<f64, IxDyn>, target_shape : &[usize]) -> Array<f64, IxDyn> {
    let mut g = grad.clone();

    while g.ndim() > target_shape.len() {
        g = g.sum_axis(Axis(0));
    }

    for (i, dim) in target_shape.iter().enumerate() {
        if *dim == 1 && g.shape()[i] > 1 {
            g = g.sum_axis(Axis(1)).insert_axis(Axis(i));
        }
    }

    g
}


#[cfg(test)]
mod tests {
    use super::Tensor;

    fn approx_eq(a: f64, b: f64, eps: f64) {
        assert!(
            (a - b).abs() <= eps,
            "values differ: left={a}, right={b}, eps={eps}"
        );
    }

    #[test]
    fn tanh_forward_matches_std() {
        let x = Tensor::from(vec![0.7]);
        let y = x.clone().tanh();

        approx_eq(y.data()[0], 0.7_f64.tanh(), 1e-12);
    }

    #[test]
    fn tanh_backward_matches_derivative() {
        let x = Tensor::from(vec![0.7]).tanh();
        let y = x.clone().tanh().mul(vec![3.0]);
        y.backward();

        let t = 0.7_f64.tanh();
        let t2 = t.tanh();
        let expected = 3.0 * (1.0 - t2 * t2);
        let got = x.inner.borrow().grad[0];

        approx_eq(got, expected, 1e-10);
    }
}
