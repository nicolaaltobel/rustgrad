use ndarray::Array;
use rustgrad::{Mlp, Tensor};
use rustgrad::Activation::Tanh;
use rustgrad::Init::{He};

fn main() {
    let mlp = Mlp::new(&[3, 4, 4, 1], He, Tanh);
    let xs = [vec![2.0, 3.0, -1.0],
        vec![3.0, -1.0, 0.5],
        vec![0.5, 1.0, 1.0],
        vec![1.0, 1.0, -1.0]];

    let ys = [1.0, -1.0, -1.0, 1.0]; // desired targets

    for epoch in 1..=200 {
        let ypred: Vec<Tensor> = xs.iter().map(|x| mlp.forward(x)).collect();

        let loss = ys
            .iter()
            .zip(ypred.iter())
            .map(|(ygt, yout)| {
                let target = Tensor::leaf(
                    Array::from_elem((1,1), *ygt).into_dyn()
                );
                yout.sub(target).pow(2.0)
            })
            .reduce(|acc, v| acc.add(&v))
            .unwrap();

        let lr = 0.05 / (1.0 + 0.01 * f64::from(epoch));
        mlp.zero_grad();
        loss.backward();
        mlp.update(lr);

        println!("epoch: {epoch}, loss: {loss}");
    }
}
