use rustgrad::Activation::{ReLU, TanH};
use rustgrad::{MLP, Value};

fn main() {
    let mlp = MLP::new(3, &[4, 4, 1], &[ReLU, ReLU, TanH]);
    let xs = [vec![2.0, 3.0, -1.0],
        vec![3.0, -1.0, 0.5],
        vec![0.5, 1.0, 1.0],
        vec![1.0, 1.0, -1.0]];

    let ys = [1.0, -1.0, -1.0, 1.0]; // desired targets

    for epoch in 1..=200 {
        let ypred: Vec<Value> = xs.iter().flat_map(|x| mlp.forward(x)).collect();
        let loss = ys
            .iter()
            .zip(ypred.iter())
            .map(|(ygt, yout)| yout.sub(Value::leaf(*ygt)).pow(2.0))
            .reduce(|acc, v| acc.add(&v))
            .unwrap();

        let lr = 0.1 / (1.0 + 0.01 * f64::from(epoch));
        mlp.zero_grad();
        loss.backward();
        mlp.update(lr);

        println!("epoch: {epoch}, loss: {loss}");
    }
}
