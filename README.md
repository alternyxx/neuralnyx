# neuralnyx
[![Crate Version](https://img.shields.io/crates/v/neuralnyx.svg)](https://crates.io/crates/neuralnyx)
[![Documentation](https://docs.rs/neuralnyx/badge.svg)](https://docs.rs/neuralnyx)
[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

A simple neural network from scratch that can be used as a library with 
quite a simple api!  

## Usage
Let's try and map the sine function with a neural network!  
  
### Initial Code
Let's start by importing everything and creating two variables x and y.  
Note that each index of x correspond to a predicted index of y. It's also such
that the nested Vec&lt;f32&gt; is so that an input or an output can be a vector.
Though we won't need the dimensionality in this case, it's useful for other functions!

<!-- I SWEAR ILL MAKE A MORE ERGONOMIC WAY TO DO THIS ;-; -->
```rust
use neuralnyx::*;

fn main() {
    let mut x: Vec<Vec<f32>> = Vec::new();
    let mut y: Vec<Vec<f32>> = Vec::new();
}
```

Then we can create a while loop to create the sin function!
```rust
let mut i = 0.0;

while i < 6.15 {
    x.push(vec![i]);
    y.push(vec![i.sin()])

    i += 0.01
}
```
<br>  <!-- idk if other people hate these manual linebreaks but i literally cant read w/o them-->

### NeuralNet Creation
Now we can start on our neural network's architecture.  

Let's start by creating the layers of our neural network. We can do that
by creating a Vec of Layer, which is given by the crate!  

Each Layer has two fields, the amount of neurons and the activation function, 
which we can get from neuralnyx::Activation.  

Let's make two layers, one with 100 neurons and a tanh function and the output 
layer which maps our y with a linear function.
```rust
let layers = vec![
    Layer {
        neurons: 100,
        activation: Activation::Tanh,
    }, Layer {
        neurons: 1,
        activation: Activation::Linear,
    }
];
```

We can now create the structure of our neural network by using Structure, again 
given by the crate.  

It takes two other fields, batch_size and cost_function, which we get by 
neuralnyxx::CostFunction. batch_size is used to specify the amount of batches sent 
to the gpu at a time and is recommended to be set to 64 if you aren't sure.  

We can just use CostFunction::MeanSquaredError and is essentially the function we use 
to measure our neural network's performance and minimize the cost! 
```rust
let structure = Structure {
    layers: layers,
    batch_size: 64,
    cost_function: CostFunction::MeanSquaredError,
};
```

But since this case is quite simple, we can just shorthand it as
```rust
let structure = Structure {
    layers,
    ..Default::default()
};
```

And now finally, we can create our neural network! 
```rust
let mut nn = NeuralNet::new(&mut x, &mut y, structure).unwrap();
```
<br>

### Training
Now, to train our neural network, we first need to specify our options with TrainingOptions.  
The optimizer specifies how the weights are tweaked. epochs (or iterations) specify how 
many times the neural network will go over the given data. And verbose specifies whether the 
cost of an epoch will be printed. There are also many other fields (currently doesnt matter). 
But these three fields are most important to know, especially for this example.
```rust
let training = TrainingOptions {
    optimizer: Optimizer::Adam(0.001),
    epochs: 5000,
    verbose: true,
    ..Default::default()
};
```

But we probably only want to specify that we do want our training to be printed, so 
let's just ignore the other two and do instead
```rust
let training = TrainingOptions {
    verbose: true,
    ..Default::default()
};
```

And now, we can train our neural network with
```rust
nn.train(&training);
```

And there we have it! The neural network will have learnt the sine function. This can be tested 
by doing
```rust
println!("{:?}", nn.test(vec![3.14]));    // should print a value close to 0!
```

Additionally, you can also print the weights and biases of the neural network by just printing 
the neural network itself.
```rust
println!("{}", nn);
```

<br>

### Try It Yourself!
You can test exactly the above tutorial by cloning the repository at 
[https://github.com/alternyxx/neuralnyx](https://github.com/alternyxx/neuralnyx) 
and running
```
cargo run --example sine
```
<br>

Moreover, there are also other examples, most notably mnist, if you do want to check it out!  
  
Please do note that it takes quite long to train neural networks, even for the basic sine function 
example.

