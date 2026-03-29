use std::cell::RefCell;
use std::collections::HashSet;
use std::fmt::{Debug, Display, Formatter};
use std::rc::Rc;

pub trait IntoValue {
    fn into_value(self) -> Value;
}

/// The inner fields of a `Value`.
struct ValueInner {
    data: f64,
    grad: f64,
    backward: Rc<dyn Fn()>,
    prev: Vec<Value>,
}

/// The type used for operations inside the model.
/// It allows to keep track of the operations and children that generated the value,
/// which is needed for backpropagation.
#[derive(Clone)]
pub struct Value {
    inner: Rc<RefCell<ValueInner>>,
}

impl Value {
    /// Creates a `Value` given its data and vector of children.
    pub(crate) fn new(data: f64, children: Vec<Value>) -> Value {
        Value {
            inner: Rc::new(RefCell::new(ValueInner {
                data,
                grad: 0.0,
                prev: children,
                backward: Rc::new(|| {}),
            })),
        }
    }

    /// Creates a `Value` with no children, given its data.
    #[must_use]
    pub fn leaf(data: f64) -> Value {
        Value {
            inner: Rc::new(RefCell::new(ValueInner {
                data,
                grad: 0.0,
                prev: Vec::new(),
                backward: Rc::new(|| {}),
            })),
        }
    }
    
    /// Maps an array of `f64` into an array of `Value` with no children.
    #[must_use]
    pub fn from(x: &[f64]) -> Vec<Value> {
        x.iter().map(|x| Value::leaf(*x)).collect()
    }
    
    /// Hyperbolic tangent function for type `Value`.
    pub(crate) fn tanh(self) -> Value {
        let x = self.inner.borrow().data;
        let t = x.tanh();
        let out = Value::new(t, vec![self.clone()]);

        let out_weak = Rc::downgrade(&out.inner);
        let self_clone = self.clone();

        let backward = {
            move || {
                self_clone.inner.borrow_mut().grad +=
                    (1.0 - t.powf(2.0))
                        * out_weak
                            .upgrade()
                            .expect("tanh backward graph node dropped")
                            .borrow()
                            .grad;
            }
        };

        out.inner.borrow_mut().backward = Rc::new(backward);

        out
    }
    
    /// Performs the rectified Linear Unit on the `data` of the `Value`.
    /// # Panics
    ///  If the upgrade of the `Weak` pointer to an `Rc` fails.
    #[must_use] 
    pub fn relu(self) -> Value {
        let x = self.inner.borrow().data;
        let out = Value::new(x.max(0.0), vec![self.clone()]);

        let out_weak = Rc::downgrade(&out.inner);
        let self_clone = self.clone();

        let backward = {
            move || {
                self_clone.inner.borrow_mut().grad += if x > 0.0 {
                    out_weak.upgrade().expect("Failed to upgrade Rc").borrow().grad
                } else {
                    0.0
                };
            }
        };
        out.inner.borrow_mut().backward = Rc::new(backward);

        out
    }

    /// Performs a topological ordering of the sequence of `Value` that generated `Self`
    /// and then recursively calls itself on the ordered vector, performing backpropagation of the gradient.
    pub fn backward(&self) {
        let mut topo: Vec<Value> = Vec::new();
        let mut visited: HashSet<*const RefCell<ValueInner>> = HashSet::new();
        let mut stack: Vec<(Value, bool)> = vec![(self.clone(), false)];

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

        self.inner.borrow_mut().grad = 1.0;

        for v in topo.iter().rev() {
            let backward = v.inner.borrow().backward.clone();
            backward();
        }
    }
    
    /// Performs addition of `Self` with a type that implements `IntoValue`, which include `Value`, `&Value`, `f64`.
    /// # Panics
    /// If the upgrade of the `Weak` pointer to an `Rc` fails.
    #[must_use]
    pub fn add(&self, other: impl IntoValue) -> Value {
        let other = other.into_value();
        let out = Value::new(
            self.inner.borrow().data + other.inner.borrow().data,
            vec![self.clone(), other.clone()],
        );

        let out_weak = Rc::downgrade(&out.inner);
        let self_clone = self.clone();
        let other_clone = other.clone();

        let backward = {
            move || {
                let out_grad = out_weak.upgrade().unwrap().borrow().grad;
                self_clone.inner.borrow_mut().grad += 1.0 * out_grad;
                other_clone.inner.borrow_mut().grad += 1.0 * out_grad;
            }
        };
        out.inner.borrow_mut().backward = Rc::new(backward);

        out
    }
    
    /// Performs multiplication of `Self` with a type that implements `IntoValue`, which include `Value`, `&Value`, `f64`.
    /// # Panics
    /// If the upgrade of the `Weak` pointer to an `Rc` fails.
    #[must_use]
    pub fn mul(&self, other: impl IntoValue) -> Value {
        let other = other.into_value();
        let out = Value::new(
            self.inner.borrow().data * other.inner.borrow().data,
            vec![self.clone(), other.clone()],
        );

        let out_weak = Rc::downgrade(&out.inner);
        let self_clone = self.clone();
        let other_clone = other.clone();

        let backward = {
            move || {
                let out_grad = out_weak.upgrade().unwrap().borrow().grad;
                self_clone.inner.borrow_mut().grad += out_grad * other_clone.inner.borrow().data;
                other_clone.inner.borrow_mut().grad += out_grad * self_clone.inner.borrow().data;
            }
        };
        out.inner.borrow_mut().backward = Rc::new(backward);

        out
    }
    
    /// Performs subtraction of `Self` with a type that implements `IntoValue`, which include `Value`, `&Value`, `f64`.
    /// # Panics
    /// If the upgrade of the `Weak` pointer to an `Rc` fails.
    #[must_use]
    pub fn sub(&self, other: impl IntoValue) -> Value {
        let other = other.into_value();
        let out = Value::new(
            self.inner.borrow().data - other.inner.borrow().data,
            vec![self.clone(), other.clone()],
        );

        let out_weak = Rc::downgrade(&out.inner);
        let self_clone = self.clone();
        let other_clone = other.clone();

        out.inner.borrow_mut().backward = Rc::new(move || {
            let out_grad = out_weak.upgrade().unwrap().borrow().grad;
            self_clone.inner.borrow_mut().grad += out_grad;
            other_clone.inner.borrow_mut().grad -= out_grad;
        });

        out
    }
    
    /// Performs the power of `Self` with an exponent whose type implements `IntoValue`, which include `Value`, `&Value`, `f64`.
    /// # Panics
    /// If the upgrade of the `Weak` pointer to an `Rc` fails.
    #[must_use]
    pub fn pow(&self, exp: f64) -> Value {
        let x = self.inner.borrow().data;
        let out = Value::new(x.powf(exp), vec![self.clone()]);

        let out_weak = Rc::downgrade(&out.inner);
        let self_clone = self.clone();

        out.inner.borrow_mut().backward = Rc::new(move || {
            let out_grad = out_weak.upgrade().unwrap().borrow().grad;
            self_clone.inner.borrow_mut().grad += exp * x.powf(exp - 1.0) * out_grad;
        });

        out
    }

    /// Sets the gradient to `x`
    pub fn set_grad(&self, x: f64) {
        self.inner.borrow_mut().grad = x;
    }

    /// Updates `data` by `- lr * grad`
    pub fn update(&self, lr: f64) {
        let grad = self.inner.borrow_mut().grad;
        self.inner.borrow_mut().data -= lr * grad;
    }

    /// Returns the value of `data`
    #[must_use] 
    pub fn data(&self) -> f64 {
        self.inner.borrow().data
    }
}

impl IntoValue for Value {
    fn into_value(self) -> Value {
        self
    }
}

impl IntoValue for &Value {
    fn into_value(self) -> Value {
        self.clone()
    }
}

impl IntoValue for f64 {
    fn into_value(self) -> Value {
        Value::leaf(self)
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Value({})", self.inner.borrow().data)
    }
}

impl Debug for Value {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Value({})", self.inner.borrow().data)
    }
}

#[cfg(test)]
mod tests {
    use super::Value;

    fn approx_eq(a: f64, b: f64, eps: f64) {
        assert!(
            (a - b).abs() <= eps,
            "values differ: left={a}, right={b}, eps={eps}"
        );
    }

    #[test]
    fn tanh_forward_matches_std() {
        let x = Value::leaf(0.7);
        let y = x.clone().tanh();

        approx_eq(y.data(), 0.7_f64.tanh(), 1e-12);
    }

    #[test]
    fn tanh_backward_matches_derivative() {
        let x = Value::leaf(0.3);
        let y = x.clone().tanh().mul(3.0);
        y.backward();

        let t = 0.3_f64.tanh();
        let expected = 3.0 * (1.0 - t * t);
        let got = x.inner.borrow().grad;

        approx_eq(got, expected, 1e-10);
    }
}
