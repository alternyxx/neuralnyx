use std::fs::File;
use std::io::Write;
use std::io::BufReader;
use neuralnyx::{NeuralNet, NeuralNetOptions, TrainingOptions};

fn main() -> std::io::Result<()> {
    // read the json file 
    let dataset = File::open("./examples/datasets/dataset.json")?;
    let reader = BufReader::new(dataset);
    let data: serde_json::Value = serde_json::from_reader(reader)?;
    
    // reorder the json to a collection of vectors
    let mut inputs: Vec<Vec<f32>> = Vec::new();
    let mut outputs: Vec<Vec<f32>> = Vec::new();

    for (board, optimal_move) in data.as_object().unwrap() {
        inputs.push(board.chars().map(|c| c.to_digit(10).unwrap() as f32).collect::<Vec<f32>>());
        // inputs.push(vec![board.parse::<f32>().unwrap()]);

        let mut output_vec = vec![0.0f32; 9]; // one hot encode the outputs
        output_vec[optimal_move.as_u64().unwrap() as usize] = 1.0;
        outputs.push(output_vec);
    }
    
    // create the neural network with the inputs and outputs
    let nn = NeuralNet::new(&mut inputs, &mut outputs, &[12], &NeuralNetOptions {
        batch_size: 64,
    }).unwrap();
    nn.train(&TrainingOptions {
        learning_rate: |n| if n < 500 { 0.05 } else { 0.01 },
        iterations: 100000,
    });

    let mut a = File::create("debug.wgsl").expect("creation failed");
    a.write(nn.generate_wgsl().as_bytes()).expect("failed");

    Ok(())
}