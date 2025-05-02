use crate::types;
use crate::generation;

use indoc::formatdoc;
use textwrap::indent;
use std::collections::HashMap;

use types::{
    Layer,
    Activation,
    CostFunction,
};
use generation::*;

pub struct Structure {
    pub batch_size: usize,   // the amount of batches sent to the gpu per compute
    // pub layers: &'static [Layer],  // documenting the pain
    pub layers: Vec<Layer>,  // layers of the neuralnet
    pub cost_function: CostFunction,  // cost function, available are mean_squared_error and cross entropy
}

impl Default for Structure {
    fn default() -> Self {
        Self {
            batch_size: 64,
            layers: Vec::new(),
            cost_function: CostFunction::MeanSquaredError,
        }
    }
}

impl Structure {
    pub(crate) fn validate(&self, n_outputs: u32) 
    -> Result<(), String> {
        let n_layers = self.layers.len();

        if self.layers.is_empty() || self.layers[n_layers - 1].neurons != n_outputs {
            return Err("The neurons of the last layer must equal to outputs".to_string());
        }

        // these checks are things currently not available
        if self.batch_size > 1024 {
            return Err("Currently not allowing batch_size greater than 1024".to_string());
        }
        
        for (i, layer) in self.layers.iter().enumerate() {
            if layer.activation == Activation::Softmax && (
                self.cost_function != CostFunction::CrossEntropy || i != n_layers - 1
            ) {
                return Err("Softmax function only supports CCE and last layer atm".to_string());   
            }
        }

        if self.cost_function == CostFunction::CrossEntropy
            && self.layers[n_layers - 1].activation != Activation::Softmax 
        {
            return Err("CCE only supports softmax outputs atm".to_string());   
        }

        Ok(())
    }

    // dynamic code generation
    // to see an example of this function, uncomment the println from template_wgsl()
    // or js try using this function after defining a structure
    pub fn generate_wgsl(&self, n_inputs: usize) -> String {
        let n_layers = self.layers.len();
        
        let mut i_weights = String::new();
        let mut i_biases = String::new();
        let mut o_weights = String::new();
        let mut o_biases = String::new();
        let mut storage = String::new();
        let mut forward = String::new();
        let mut backpropagate = String::new();
        let mut atomic_averaging = String::new();

        let mut prev_layer_outputs = n_inputs as u32;

        // for backpropagate code generation
        let mut reversed_layers = self.layers.to_vec();
        let mut next_layer_inputs = reversed_layers.pop().unwrap().neurons;
        reversed_layers = reversed_layers.into_iter().rev().collect::<Vec<Layer>>();
        reversed_layers.push(Layer {
            neurons: prev_layer_outputs,
            activation: Activation::Linear,
        });

        // the zero should never happen since
        // the variable wouldnt be used on first iteration
        let mut next_next_layer_inputs = 0;

        // purposefully not an iterator loop since the formatting gets ugly
        // it is also worth noting that idk how to seperate this to different functions
        // without making a billion parameters so its js clustered here
        // could technically create a class that is used every iteration but uhhhhhh... dk
        for i in 0..n_layers {
            let decrement = n_layers - i - 1;

            let neurons = self.layers[i].neurons;
            let reverse_n_neurons = reversed_layers[i].neurons;

            i_weights += &format!("    weights{i}: array<array<float, {prev_layer_outputs}>, {neurons}>,\n");
            i_biases += &format!("    biases{i}: array<float, {neurons}>,\n");
            o_weights += &format!("    weights{i}: array<array<atomic<u32>, {prev_layer_outputs}>, {neurons}>,\n");
            o_biases += &format!("    biases{i}: array<atomic<u32>, {neurons}>,\n");

            storage += &formatdoc! {"
                var<private> al{i}: array<f32, {neurons}>;
                var<private> delta{i}: array<f32, {neurons}>;
            "};

            let forward_input = if i != 0 {
                &format!("al{}", i - 1)
            } else { "X[id.x]" };
            
            
            let activation_function = self.layers[i].activation;
            let (mut softmax_activation, mut activation) = (String::new(), String::new());
            
            // softmax requires special care since it requires sum
            if activation_function == Activation::Softmax {
                softmax_activation = format!("{activation_function}();");
            } else {
                activation = format!("al{i}[i] = {activation_function}(al{i}[i]);");
            }

            // i did try to use the indent macro but- issues... 
            // and i still really wanna make it look pretty
            forward += &indent(&formatdoc! {"
                for (var i = 0; i < {neurons}; i++) {{
                    al{i}[i] = 0.0;
                    for (var j = 0; j < {prev_layer_outputs}; j++) {{
                        al{i}[i] += weights.weights{i}[i][j] * {forward_input}[j];
                    }}
                    al{i}[i] += biases.biases{i}[i];
                    {activation}
                }};
                {softmax_activation}
            "}, "    ");
            
            // ~~~ backpropagation code generation! ~~~
            let backpropagation_input = if decrement != 0 {
                &format!{"al{}", decrement - 1}
            } else { "X[id.x]" };

            let grad_weights_cas = generate_cas(
                &format!("outputs.grad_weights.weights{decrement}[i][j]"),
                &format!("{backpropagation_input}[j] * tmp"),
                "        ",
            );

            let grad_biases_cas = generate_cas(
                &format!("outputs.grad_biases.biases{decrement}[i]"),
                "tmp",
                "    ",
            );

            // the difference between the two is that if its the last layer of backprog, we calculate the delta
            // as just dal^L/dzl^L dC/dal^L whereas if not, its the sum((W^L)_i,j delta^L+1) (I THINK) 
            if i == 0 {
                let n_al = n_layers - 1;

                let last_layer_activation = self.layers[n_layers - 1].activation;

                // check below is kinda redundant rn since this is done in validation
                // but its kinda important for da future
                let delta = if last_layer_activation == Activation::Softmax
                    && self.cost_function == CostFunction::CrossEntropy
                {
                    format!("(al{n_al}[i] - targets[id.x][i])")
                } else {
                    format!("d{last_layer_activation}(al{n_al}[i]) * 2.0 * (al{n_al}[i] - targets[id.x][i])")
                };

                backpropagate += &indent(&formatdoc! {"
                    for (var i = 0; i < {next_layer_inputs}; i++) {{
                        let tmp = {delta};
                        
                        for (var j = 0; j < {reverse_n_neurons}; j++) {{
                            {grad_weights_cas}
                        }}
                        
                        {grad_biases_cas}
                        delta{decrement}[i] = tmp;
                    }}
                "}, "    ");
            } else {
                let next_layer = decrement + 1;
                let activation_function = reversed_layers[i - 1].activation;

                backpropagate += &indent(&formatdoc! {"
                    for (var i = 0; i < {next_layer_inputs}; i++) {{
                        var sum = 0.0;
                        for (var j = 0; j < {next_next_layer_inputs}; j++) {{
                            sum += weights.weights{next_layer}[j][i] * delta{next_layer}[j];
                        }}

                        let tmp = sum * d{activation_function}(al{decrement}[i]);

                        for (var j = 0; j < {reverse_n_neurons}; j++) {{
                            {grad_weights_cas}
                        }}

                        {grad_biases_cas}
                        delta{decrement}[i] = tmp;
                    }}
                "}, "    ");
            }

            atomic_averaging += &indent(&formatdoc! {"
                for (var i = 0; i < {neurons}; i++) {{
                    for (var j = 0; j < {prev_layer_outputs}; j++) {{
                        atomicStore(
                            &outputs.grad_weights.weights{i}[i][j],
                            bitcast<u32>(
                                bitcast<f32>(atomicLoad(&outputs.grad_weights.weights{i}[i][j])) / float_batch_size
                            )
                        );
                    }}
                    atomicStore(
                        &outputs.grad_biases.biases{i}[i],
                        bitcast<u32>(
                            bitcast<f32>(atomicLoad(&outputs.grad_biases.biases{i}[i])) / float_batch_size
                        )
                    );
                }}
            "}, "        ");

            prev_layer_outputs = neurons;
            next_next_layer_inputs = next_layer_inputs;
            next_layer_inputs = reverse_n_neurons;
        }

        Self::template_wgsl(include_str!("neuralnet.wgsl").into(), &HashMap::from([
            ("batch_size".to_string(), self.batch_size.to_string()),
            ("n_inputs".to_string(), n_inputs.to_string()),
            ("n_outputs".to_string(), prev_layer_outputs.to_string()),
            ("n_al".to_string(), (n_layers - 1).to_string()),
            ("i_weights".to_string(), i_weights),
            ("i_biases".to_string(), i_biases),
            ("o_weights".to_string(), o_weights),
            ("o_biases".to_string(), o_biases),
            ("storage".to_string(), storage),
            ("cost_function".to_string(), self.cost_function.to_string()),
            ("forward".to_string(), forward),
            ("backpropagate".to_string(), backpropagate),
            // ("atomic_averaging".to_string(), atomic_averaging),
        ])).into()
    }

    /*
        this function is created because i want js/ts template literals and
        pipeline constants aren't enough
        also im not gonna write a whole parser just so this can ignore comments xd
        it is also worth nothing that it is VERY lazily made
    */
    fn template_wgsl(wgsl: &str, literals: &HashMap<String, String>) -> String {
        let mut templating = false;
        let mut template_variable: String = String::new();
        let mut templated_wgsl: String = String::new();

        let mut chars = wgsl.chars().peekable();
        while let Some(char) = chars.next() {
            // in the process of templating
            if templating {
                if char == '}' {
                    templated_wgsl += literals.get(&template_variable.to_string())
                        .unwrap_or_else(|| panic!("\n{} wasn't given\n", template_variable.to_string()));

                    template_variable = String::new();
                    templating = false;
                } else {
                    template_variable += &char.to_string();    
                }

                continue
            } else if char == '$' {
                // i tried finding a clever solution with next_if() :C
                if chars.peek() == Some(&'{') {
                    chars.next();
                    templating = true;
                }
            } else {
                templated_wgsl += &char.to_string();
            }
        }
        
        println!("{templated_wgsl}"); // lazy debugging :P
        templated_wgsl
    }
}

// logic seems simple enough so ill js put it here :P
impl Activation {
    pub(crate) fn activate(&self, zl: &mut Vec<f32>) {
        match self {
            Activation::Linear => {},
            Activation::Tanh => {
                for i in 0..zl.len() {
                    zl[i] = zl[i].tanh();
                }
            },
            Activation::Relu => {
                for i in 0..zl.len() {
                    zl[i] = zl[i].max(0.0);
                }
            },
            Activation::Sigmoid => {
                for i in 0..zl.len() {
                    zl[i] = 1.0 / (1.0 + (-zl[i]).exp());
                }
            },
            Activation::Softmax => {
                let n_logits = zl.len();

                let mut highest = zl[0];
                for i in 1..n_logits {
                    highest = zl[i].max(highest);
                }
                
                // calculate e^zl_i
                let mut sum = 1.0e-20;
                for i in 0..n_logits {
                    let tmp = (zl[i] - highest).exp();
                    zl[i] = tmp;
                    sum += tmp; 
                }
            
                // e^zl_i / sum(e^zl)
                for i in 0..n_logits {
                    zl[i] /= sum;
                }
            },
        }
    }
}
