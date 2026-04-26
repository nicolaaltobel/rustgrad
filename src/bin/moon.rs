use ndarray::Array;
use rand::rngs::StdRng;
use rand::Rng;
use rustgrad::{Activation::Tanh, Init::He, Mlp, Tensor};
use std::fmt::Write as _;
use ndarray_rand::rand::SeedableRng;

const DATA_SEED: u64 = 1337;
const GRID_STEPS: usize = 100;

fn main() {
    let mlp = Mlp::new(&[2, 64, 32, 16, 1], He, Tanh);
    let (xs, ys) = make_moons(4000, DATA_SEED);
    let n_train = 3200;

    train(&mlp, &xs, &ys, n_train, 500);

    let preds: Vec<f64> = xs
        .iter()
        .map(|x| mlp.forward(x).data()[[0, 0]])
        .collect();

    save_grid(&mlp, &xs, &ys, &preds, n_train, "src/bin/plots/moons.csv");
}

#[allow(clippy::cast_precision_loss)]
fn make_moons(n: usize, seed: u64) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let half = n / 2;
    let mut xs = Vec::new();
    let mut ys = Vec::new();

    for i in 0..half {
        let angle = std::f64::consts::PI * i as f64 / half as f64;
        xs.push(vec![
            angle.cos() + rng.random_range(-0.3..0.3),
            angle.sin() + rng.random_range(-0.3..0.3),
        ]);
        ys.push(1.0);
    }

    for i in 0..half {
        let angle = std::f64::consts::PI * i as f64 / half as f64;
        xs.push(vec![
            1.0 - angle.cos() + rng.random_range(-0.3..0.3),
            0.5 - angle.sin() + rng.random_range(-0.3..0.3),
        ]);
        ys.push(-1.0);
    }

    let mut indices: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = rng.random_range(0..=i);
        indices.swap(i, j);
    }

    let xs = indices.iter().map(|&i| xs[i].clone()).collect();
    let ys = indices.iter().map(|&i| ys[i]).collect();

    (xs, ys)
}

#[allow(clippy::cast_precision_loss)]
fn loss(ys: &[f64], ypred: &[Tensor]) -> Tensor {
    ys.iter()
        .zip(ypred.iter())
        .map(|(y, ypred)| {
            let target = Tensor::leaf(Array::from_elem((1, 1), *y).into_dyn());
            let margin = ypred.mul(target);
            let one = Tensor::leaf(Array::from_elem((1, 1), 1.0).into_dyn());
            one.sub(margin).relu()
        })
        .reduce(|acc, v| acc.add(&v))
        .unwrap()
        .mul(Tensor::leaf(
            Array::from_elem((1, 1), 1.0 / ys.len() as f64).into_dyn(),
        ))
}

#[allow(clippy::cast_precision_loss)]
fn train(mlp: &Mlp, xs: &[Vec<f64>], ys: &[f64], train_split: usize, epochs: usize) {
    let (xs_train, xs_test) = xs.split_at(train_split);
    let (ys_train, ys_test) = ys.split_at(train_split);
    let batch_size = 32;

    let mut best_acc = 0.0;
    let patience_reset: usize = 120;
    let min_epochs = 120;
    let mut patience = patience_reset;

    for epoch in 1..=epochs {
        let lr = 0.1 / (1.0 + 0.003 * epoch as f64);
        let mut epoch_loss = 0.0;

        for start in (0..xs_train.len()).step_by(batch_size) {
            let end = (start + batch_size).min(xs_train.len());
            let batch_x = &xs_train[start..end];
            let batch_y = &ys_train[start..end];

            let ypred: Vec<Tensor> = batch_x.iter().map(|x| mlp.forward(x)).collect();
            let batch_loss = loss(batch_y, &ypred);
            epoch_loss += batch_loss.data()[[0, 0]];

            mlp.zero_grad();
            batch_loss.backward();
            mlp.update(lr);
        }

        let train_acc = accuracy(mlp, xs_train, ys_train);
        let test_acc = accuracy(mlp, xs_test, ys_test);

        if test_acc > best_acc {
            best_acc = test_acc;
            patience = patience_reset;
        } else {
            patience -= 1;
            if epoch >= min_epochs && patience == 0 {
                println!(
                    "Early stop at epoch {epoch}: best test accuracy {best_acc:.4}, lr {lr:.5}"
                );
                return;
            }
        }

        if epoch % 20 == 0 {
            println!(
                "Epoch: {epoch}, loss: {epoch_loss:.4}, train accuracy: {train_acc:.4}, test accuracy: {test_acc:.4}, lr: {lr:.5}",
            );
        }
    }
}

#[allow(clippy::cast_precision_loss)]
fn accuracy(mlp: &Mlp, xs: &[Vec<f64>], ys: &[f64]) -> f64 {
    let correct = xs
        .iter()
        .zip(ys.iter())
        .filter(|&(x, &y)| {
            let pred = mlp.forward(x).data()[[0, 0]];
            (pred > 0.0) == (y > 0.0)
        })
        .count();
    correct as f64 / ys.len() as f64
}

fn save_grid(mlp: &Mlp, xs: &[Vec<f64>], ys: &[f64], preds: &[f64], split: usize, path: &str) {
    assert_eq!(xs.len(), ys.len());
    assert_eq!(xs.len(), preds.len());

    let mut out = String::from("x1,x2,label,pred,split\n");

    for (i, (x, y)) in xs.iter().zip(ys.iter()).enumerate() {
        let s = if i < split { "train" } else { "test" };
        let _ = writeln!(out, "{},{},{},{},{}", x[0], x[1], y, preds[i], s);
    }

    #[allow(clippy::cast_precision_loss)]
    for i in 0..GRID_STEPS {
        for j in 0..GRID_STEPS {
            let x1 = -1.5 + 4.0 * j as f64 / GRID_STEPS as f64;
            let x2 = -1.0 + 3.5 * i as f64 / GRID_STEPS as f64;
            let pred = mlp.forward(vec![x1, x2]).data()[[0, 0]];
            let _ = writeln!(out, "{x1},{x2},grid,{pred},grid");
        }
    }

    std::fs::write(path, out).unwrap();
}