use std::fs::File;
use std::io::{BufReader, Read};
use flate2::read::GzDecoder;
use neuralnyx::NeuralNet;

#[derive(Debug, serde::Deserialize)]
struct Digit {
    image: Vec<f32>,
    label: usize,
}

fn main() -> std::io::Result<()> {
    let path = "./examples/datasets/mnist_handwritten_test.json.gz";
    let mnist_dataset = File::open(path)?;

    let mut json = GzDecoder::new(BufReader::new(mnist_dataset));
    let mut mnist_json = String::new();
    json.read_to_string(&mut mnist_json).unwrap();

    let mnist_dataset: Vec<Digit> = serde_json::from_str(&mnist_json).unwrap();

    let mut inputs: Vec<Vec<f32>> = Vec::new();
    let mut outputs: Vec<Vec<f32>> = Vec::new();
    for digit in mnist_dataset {
        inputs.push(digit.image);

        let mut output_vec = vec![0.0f32; 10];
        output_vec[digit.label] = 1.0;
        outputs.push(output_vec);
    }   
    
    let mut nn = NeuralNet::new(&mut inputs, &mut outputs, &[512]).unwrap();
    nn.train(0.01);

    Ok(())
}