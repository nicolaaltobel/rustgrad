use ndarray::{stack, Array, Axis, Ix2, IxDyn};
use rand::{Rng, SeedableRng};
use rand::prelude::StdRng;
use rustgrad::Activation::Tanh;
use rustgrad::Init::He;
use rustgrad::{Mlp, Tensor};

fn main(){
    let mut names : Vec<String> = Vec::new();
    let mut reader = csv::Reader::from_path("src/bin/data/names.csv").expect("cannot open names.csv");

    for entry in reader.records().flatten().filter(|s| !s.as_slice().contains(' ') && !s.is_empty()) {
            names.push(entry[0].to_string());
    }

    let mut probs = Array::<f32, _>::zeros((27,27));

    let mut xs = Vec::new();
    let mut ys = Vec::new();

    for name in names{
        let indexes = (".".to_string() + &name + ".").chars().collect::<Vec<char>>();

        for w in indexes.windows(2) {
            if let Ok(i) = ctoi(w[0]) && let Ok(j) = ctoi(w[1]){
                probs[[i,j]] += 1.;
                xs.push(one_hot(i));
                ys.push(j);
            }
        }
    }
    //println!("{probs}");

    /*for k in 1..27 {
        let mut start = itoc(k);
        let mut result = start.to_string();
        loop {
            let i = ctoi(start).expect("Fail bruh");
            let next = itoc(highest(probs.index_axis(Axis(0), i)));
            if start == '.' {
                break;
            }
            result.push(next);
            start = next;
        }
        println!("{result}");
    }

     */


    //println!("probs: {probs:?}");


    let nn = Mlp::new(&[27,27], He, Tanh);

    let x_batch = Tensor::leaf(
        stack(
            Axis(0),
            &xs.iter().map(|x| x.view()).collect::<Vec<_>>(),
        ).unwrap().into_dyn()
    );

    for epoch in 1..=100{
        let logits = nn.forward(&x_batch);
        let loss = logits.log_softmax(1).nll_loss(&ys);

        nn.zero_grad();
        loss.backward();
        nn.update(10.);

        println!("epoch {epoch}, loss: {loss}");
    }
    let mut rng = StdRng::seed_from_u64(42);
    for _ in 0..100 {
        let mut c = '.';
        let mut name = String::new();
        loop {
            let x = Tensor::leaf(
                one_hot(ctoi(c).unwrap())
                    .into_shape_with_order((1, 27))
                    .unwrap()
                    .into_dyn(),
            );
            let log_probs = nn.forward(&x).log_softmax(1);
            let data = log_probs
                .data()
                .into_dimensionality::<Ix2>()
                .unwrap();
            let probs = data.row(0).mapv(f64::exp);

            // campiona dalla distribuzione
            let r: f64 = rng.random();
            let mut cumsum = 0.0;
            let next_idx = probs
                .iter()
                .enumerate()
                .find(|(_, p)| {
                    cumsum += *p;
                    cumsum >= r
                })
                .map_or(probs.len()-1, |(i, _)| i);

            c = itoc(next_idx);
            if c == '.' {
                break;
            }
            name.push(c);
            //print!("{c}");
            //stdout().flush().unwrap();
        }
        println!("{name}");
    }
}

fn ctoi (mut c : char) -> Result<usize, String>{
    c = c.to_ascii_lowercase();
    //print!("{c}");
    if c == '.' {
        Ok(0)
    } else if c.is_ascii_alphabetic(){
        Ok(c as usize - 'a' as usize + 1)
    } else {
        Err(format!("invalid character: {c}"))
    }
}

#[allow(clippy::cast_possible_truncation, dead_code)]
fn itoc (n : usize) -> char {
    if n == 0 {
        '.'
    } else {
        (n + 'a' as usize - 1) as u8 as char
    }
}

#[allow(dead_code)]
fn highest(arr : [i32; 27]) -> usize{
    let mut max = 0;
    let mut imax = 0;

    for (i, &n) in arr.iter().enumerate(){
        if n >= max {
            max = n;
            imax = i;
        }
    }
    imax
}

fn one_hot(i : usize) -> Array<f64, IxDyn>{
    let mut out = Array::zeros(27).into_dyn();
    out[i] = 1.;

    out
}