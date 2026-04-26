use ndarray::{Array, IxDyn};
use rustgrad::Activation::Tanh;
use rustgrad::Init::He;
use rustgrad::Mlp;

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
                ys.push(one_hot(j));
            }
        }
    }
    println!("{probs}");

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