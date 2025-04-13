use std::fs::File;
use std::io::BufReader;
use neuralnyx::{
    NeuralNet,
    Structure,
    Layer,
    Activation,
    TrainingOptions,
    PreTrained,
};

fn main() -> std::io::Result<()> {
    // read the json file 
    let dataset = File::open("./examples/datasets/dataset.json")?;
    let reader = BufReader::new(dataset);   // with a buffer
    let data: serde_json::Value = serde_json::from_reader(reader)?;
    
    // reorder the json to a collection of vectors
    let mut inputs: Vec<Vec<f32>> = Vec::new();
    let mut outputs: Vec<Vec<f32>> = Vec::new();

    for (board, optimal_move) in data.as_object().unwrap() {
        inputs.push(board.chars().map(          // a move was encoded as a string so,
            |c| c.to_digit(10).unwrap() as f32  // change that to chars and then convert
        ).collect::<Vec<f32>>());               // them to a Vec<f32>

        let mut output_vec = vec![0.0f32; 9]; // one hot encode the outputs
        output_vec[optimal_move.as_u64().unwrap() as usize] = 1.0;
        outputs.push(output_vec);
    }
    
    // create the neural network with out dataset
    let mut nn = NeuralNet::new(&mut inputs, &mut outputs, Structure {
        batch_size: 64,
        layers: &[
            Layer {
                n_neurons: 9,
                activation: Activation::Softmax,    // probability distribution
            },
        ],
        ..Default::default()
    }).unwrap();

    // train the neuralnet with a custom learning rate function and 100,000 iterations
    nn.train(&TrainingOptions {
        learning_rate: |i| if i < 100000 { 0.01 } else { 0.002 },
        iterations: 100000,
    });

    println!("{:?}", nn.weights);

    let pretrained = PreTrained::from(nn);

    Ok(())
}