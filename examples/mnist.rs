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

    // download the mnist dataset thats in json format
    let mut easy = Easy::new();
    
    easy.url(
        "https://github.com/lorenmh/mnist_handwritten_json/raw/master/mnist_handwritten_train.json.gz"
    ).unwrap();
    easy.follow_location(true).unwrap();
    
    {
        let mut transfer = easy.transfer();
        transfer.write_function(|data| {
            training.extend_from_slice(data);
            Ok(data.len())
        }).unwrap();
        transfer.perform().unwrap();
    }

    easy.url(
        "https://github.com/lorenmh/mnist_handwritten_json/raw/master/mnist_handwritten_test.json.gz"
    ).unwrap();
    easy.follow_location(true).unwrap();

    {
        let mut transfer = easy.transfer();
        transfer.write_function(|data| {
            testing.extend_from_slice(data);
            Ok(data.len())
        }).unwrap();
        transfer.perform().unwrap();
    }

    let (mut mnist_training_json, mut mnist_testing_json) = (String::new(), String::new());

    GzDecoder::new(&training[..]).read_to_string(&mut mnist_training_json).unwrap(); // decoding the zipped json
    GzDecoder::new(&testing[..]).read_to_string(&mut mnist_testing_json).unwrap();

    // create digits from the decoded json string
    let (mut images, mut labels) = (Vec::new(), Vec::new());
    let mnist_dataset: Vec<Digit> = serde_json::from_str(&mnist_testing_json).unwrap();
        // .iter()
        // .map(|digit| {
        //     images.push(
        //         digit["image"].as_array().unwrap()
        //         .iter()
        //         .map(|x| x.as_f64().unwrap() as f32)
        //         .collect::<Vec<f32>>()
        //     );
            
        //     let mut label_vec = vec![0.0f32; 10]; // one hot encode the output
        //     label_vec[digit["label"].as_u64().unwrap() as usize] = 1.0;
        //     labels.push(label_vec);
        // });
    
    // reorder the digits to collection of vectors
    for digit in mnist_dataset {
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
        cost_threshold: 0.1,
        shuffle_data: true,
        verbose: true,
        ..Default::default()
    });

    let accuracy = nn.gauge(&images, &labels).unwrap();

    println!("Accuracy: {}% w/ Cost: {}", accuracy, cost);

    Ok(())
}