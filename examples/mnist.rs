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
    let (mut training, mut testing) = (Vec::new(), Vec::new());

    // download the mnist datasets thats in json format
    let mut curl = Easy::new();
    
    curl.url(
        "https://github.com/lorenmh/mnist_handwritten_json/raw/master/mnist_handwritten_train.json.gz"
    ).unwrap();
    curl.follow_location(true).unwrap();
    
    {
        println!("Downloading the training dataset...");
        let mut transfer = curl.transfer();
        transfer.write_function(|data| {
            training.extend_from_slice(data);
            Ok(data.len())
        }).unwrap();
        transfer.perform().unwrap();
    }

    curl.url(
        "https://github.com/lorenmh/mnist_handwritten_json/raw/master/mnist_handwritten_test.json.gz"
    ).unwrap();
    curl.follow_location(true).unwrap();

    {
        println!("Downloading the testing dataset...");
        let mut transfer = curl.transfer();
        transfer.write_function(|data| {
            testing.extend_from_slice(data);
            Ok(data.len())
        }).unwrap();
        transfer.perform().unwrap();
    }

    let (mut mnist_training_json, mut mnist_testing_json) = (String::new(), String::new());

    // decoding the zipped jsons
    GzDecoder::new(&training[..]).read_to_string(&mut mnist_training_json).unwrap();
    GzDecoder::new(&testing[..]).read_to_string(&mut mnist_testing_json).unwrap();

    // create digits from the decoded json string
    let (mut images, mut labels) = (Vec::new(), Vec::new());
    let mnist_training_dataset: Vec<Digit> = serde_json::from_str(&mnist_training_json).unwrap();
    
    // reorder the digits to collection of vectors
    for digit in mnist_training_dataset {
        images.push(digit.image);

        let mut output_vec = vec![0.0f32; 10]; // one hot encode the output
        output_vec[digit.label] = 1.0;
        labels.push(output_vec);
    }
    
    // // create and train the neural network with the images and labels
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
        shuffle_data: true,
        cost_threshold: 0.1,
        verbose: true,
        ..Default::default()
    });

    println!("Training complete! Beginning testing...");

    // Structure our testing data for the trained neural network
    let (mut images, mut labels) = (Vec::new(), Vec::new());
    let mnist_testing_dataset: Vec<Digit> = serde_json::from_str(&mnist_testing_json).unwrap();
    for digit in mnist_testing_dataset {
        images.push(digit.image);

        let mut output_vec = vec![0.0f32; 10];
        output_vec[digit.label] = 1.0;
        labels.push(output_vec);
    }

    let accuracy = nn.gauge(&images, &labels).unwrap();

    println!("Accuracy: {}% w/ Cost: {}", accuracy, cost);

    Ok(())
}