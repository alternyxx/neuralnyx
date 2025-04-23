use neuralnyx::*;

fn main() {
    // create our sine function and push them to x and y
    let mut x: Vec<Vec<f32>> = Vec::new();
    let mut y: Vec<Vec<f32>> = Vec::new();

    let mut i = 0.0;

    while i < 6.15 {
        x.push(vec![i]);
        y.push(vec![i.sin()]);
    
        i += 0.01;
    }

    // a tanh layer with 100 neurons and linear output layer
    let layers = vec![
        Layer {
            neurons: 100,
            activation: Activation::Tanh,
        }, Layer {
            neurons: 1,
            activation: Activation::Linear,
        },
    ];
    
    // create the strucute of the neural network
    let structure = Structure {
        layers,
        ..Default::default()    
    };

    // actually create the neural network
    let mut nn = NeuralNet::new(&mut x, &mut y, structure).unwrap();

    // specify that we want the training to be printed
    let training = TrainingOptions {
        verbose: true,
        ..Default::default()
    };

    // train the neural network
    nn.train(training);

    // run some tests
    println!("sin(pi/2) ≈ {:?}", nn.test(vec![1.57]));    // should be close to 1
    println!("sin(pi) ≈ {:?}", nn.test(vec![3.14]));    // should be close to 0
}