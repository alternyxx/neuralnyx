use std::fs::File;
use std::io::{BufReader, Read};
use flate2::read::GzDecoder;

use neuralnyx::{
    NeuralNet,
    Structure,
    Layer,
    TrainingOptions,
};
use neuralnyx::Activation::{Relu, Softmax};
use neuralnyx::CostFunction::CrossEntropy;
use neuralnyx::Optimizer::Adam;

#[derive(Debug, serde::Deserialize)]
struct Digit {
    image: Vec<f32>,
    label: usize,
}

fn main() -> std::io::Result<()> {
    // read the json file
    let path = "./examples/datasets/mnist_handwritten_test.json.gz";
    let mnist_dataset = File::open(path)?;

    let mut json = GzDecoder::new(BufReader::new(mnist_dataset)); // decoding the zipped json
    let mut mnist_json = String::new();
    json.read_to_string(&mut mnist_json).unwrap();

    // create digits from the decoded json string
    let mnist_dataset: Vec<Digit> = serde_json::from_str(&mnist_json).unwrap();

    // reorder the digits to collection of vectors
    let mut images: Vec<Vec<f32>> = Vec::new();
    let mut labels: Vec<Vec<f32>> = Vec::new();
    
    for digit in mnist_dataset {
        images.push(digit.image);

        let mut output_vec = vec![0.0f32; 10]; // one hot encode the output
        output_vec[digit.label] = 1.0;
        labels.push(output_vec);
    }   
    
    // create and train the neural network with the images and labels
    let mut nn = NeuralNet::new(&mut images, &mut labels, Structure {
        layers: vec![
            Layer {
                neurons: 512,
                activation: Relu,
            }, Layer {
                neurons: 10,
                activation: Softmax,
            },
        ],
        cost_function: CrossEntropy,
        ..Default::default()
    }).unwrap();

    nn.train(&TrainingOptions {
        optimizer: Adam(0.001),
        epochs: 30,
        verbose: true,
        ..Default::default()
    });

    Ok(())
}