use curl::easy::Easy;
use std::io::Read;
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
    let mut training: Vec<u8> = Vec::new();

    {
        let mut easy = Easy::new();
        easy.url(
            "https://github.com/lorenmh/mnist_handwritten_json/raw/master/mnist_handwritten_test.json.gz"
        ).unwrap();
        
        easy.follow_location(true).unwrap();

        let mut transfer = easy.transfer();
        transfer.write_function(|data| {
            training.extend_from_slice(data);
            Ok(data.len())
        }).unwrap();
        
        transfer.perform().unwrap();
    }

    let mut json = GzDecoder::new(&training[..]); // decoding the zipped json
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
                neurons: 128,
                activation: Relu,
            }, Layer {
                neurons: 10,
                activation: Softmax,
            },
        ],
        cost_function: CrossEntropy,
        ..Default::default()
    }).unwrap();

    let cost = nn.train(TrainingOptions {
        optimizer: Adam(0.001),
        epochs: 10,
        cost_threshold: 0.1,
        shuffle_data: true,
        verbose: true,
        ..Default::default()
    });
    
    let accuracy = nn.gauge(&images, &labels).unwrap();

    println!("Accuracy: {}% w/ Cost: {}", accuracy, cost);

    Ok(())
}