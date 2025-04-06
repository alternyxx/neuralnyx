# neuralnet
A simple neural network from scratch.

### Usage
All you need to train a neural network is just 
```rust
use neuralnyx::NeuralNet;

let mut nn = NeuralNet::new(&mut inputs, &mut outputs, &layers).unwrap();
nn.train(learning_rate);
```
whereby inputs is a Vec\<Vec\<f32\>\>, which we can think of as a collection of input vectors
and outputs is also a Vec\<Vec\<f32\>\>, which we can also think of as a collection of 
corresponding output vectors. layers is a [i32] basically, each element of layers describe
the amount of neurons in that layer.  
And finally, learning_rate is just an f32 and self-explanatory.
This is right now just a prototype and i might switch to a struct for the inputs and outputs 
to describe correspondance later down the line.

### Why?
idk- i shouldve prob js used a framework- ;-; this did NOT need any gpu operations

# Explanation
The neural network is divided into two parts, neuralnet.rs and neuralnet.wgsl  
The forward pass is done in neuralnet.wgsl  
and we retrieve back the data and do the backward propagation in the rust side.
(maybe, im thinking about this)

The wgsl side is generated via a template function since all shading languages dont allow
functions that accepts array of runtime-size so we just manually generate them. 

# To-Do / Bugs
There's an issue where the weights or biases just turn to NaNs. This'll probably persist for a while since
I am too lazy to debug around but we'll see.
