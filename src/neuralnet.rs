use crate::pipeline;
use crate::utils;
use crate::types;

use rand::prelude::*;
use rand_distr::Normal;
use indoc::formatdoc;
use textwrap::indent;
use pollster::FutureExt;
use std::collections::HashMap;
use std::mem::size_of;
use pipeline::NeuralNetPipeline;
use utils::{flatten2d, flatten3d, template_wgsl};
use types::{
    Layer,
    Activation,
    Structure,
    TrainingOptions,
};

// currently i do NOT want to deal with lifetimes :/
pub struct NeuralNet {
    pipeline: NeuralNetPipeline,    // the gpu pipeline and its functions
    batches: Vec<Vec<Vec<f32>>>,        // for these two, the inner vec is the vector inputs and outputs,
    targets: Vec<Vec<Vec<f32>>>,        // for middle vec is a single batch, and the outer vec groups the batches
    pub weights: Vec<Vec<Vec<f32>>>,    // for this, we can consider the outer vec as the layers and the two inner as a matrix
    pub biases: Vec<Vec<f32>>,  // outer vec is layers, inner vec is the vector of biases :/
    pub(crate) structure: Structure,
}

impl NeuralNet {
    // more direct approach to create NeuralNet
    pub fn new(
        inputs: &mut Vec<Vec<f32>>, 
        outputs: &mut Vec<Vec<f32>>, 
        structure: Structure,
    ) -> Result<NeuralNet, String> {
        // ~~~ checks to ensure variables are valid ~~~
        // putting it in a seperate function is ugly but ts ugly too, pmo
        let n_input_v: usize;
        if inputs.is_empty() {
            return Err("there's nothing to train on?...".to_string());
        } else {
            n_input_v = inputs[0].len();
            for input in inputs.iter() {
                if input.len() != n_input_v {
                    return Err(
                        "number of elements in input vectors must stay consistent"
                            .to_string()
                    );
                }
            }
        }
        
        let n_output_v: usize;
        if outputs.is_empty() {
            return Err(
                "uhh- i mean likee? i guess technically?... but u might as well just- coinflip"
                    .to_string()
            );
        } else {
            n_output_v = outputs[0].len();
            for output in outputs.iter() {
                if output.len() != n_output_v {
                    return Err(
                        "number of elements in output vectors must stay consistent"
                            .to_string()
                    );
                }
            }
        }

        if inputs.len() != outputs.len() {
            return Err("number of inputs must map to number of outputs"
                .to_string()
            );
        }

        // the gpu pipeline
        let pipeline = NeuralNetPipeline::new().block_on();

        let (weights, biases) = NeuralNet::initialize(n_input_v, structure.layers); // weight and biases initialization

        // ~~~ seperate the inputs into batches ~~~
        let batches: Vec<Vec<Vec<f32>>> = inputs
            .chunks(structure.batch_size)
            .map(|s| s.to_vec())
            .collect();
        let targets: Vec<Vec<Vec<f32>>> = outputs
            .chunks(structure.batch_size)
            .map(|s| s.to_vec())
            .collect();

        Ok(Self {
            pipeline,
            batches,
            targets,
            weights,
            biases,
            structure,    
        })
    }

    fn initialize(n_input_v: usize, layers: &[Layer]) 
    -> (Vec<Vec<Vec<f32>>>, Vec<Vec<f32>>) {
        let mut rng = rand::rng();

        let mut weights: Vec<Vec<Vec<f32>>> = Vec::new();
        let mut biases: Vec<Vec<f32>> = Vec::new();
        
        let mut n_prev_outputs = n_input_v as u32;
        for layer in layers.iter() {
            // a weights matrix or a 2d vector 
            weights.push((0..layer.n_neurons)
                .map(|_| (0..n_prev_outputs)
                .map(|_| {
                    Normal::new(0.0, (2.0 / n_prev_outputs as f32).sqrt()) // kaiming initialization
                        .unwrap().sample(&mut rng) 
                }).collect()
            ).collect::<Vec<Vec<f32>>>());

            biases.push(vec![0.01; layer.n_neurons as usize]); // just so relu doesnt cause problems

            n_prev_outputs = layer.n_neurons;
        }
    
        (weights, biases)
    }

    // dynamic code generation
    // to see an example of this function, uncomment the println from template_wgsl()
    pub fn generate_wgsl(&self) -> String {
        let n_layers = self.structure.layers.len();
        let n_inputs = self.batches[0][0].len();

        // looks clearner ig
        let (mut i_weights, mut i_biases, mut storage, mut forward, mut backpropagate) = (
            String::new(),
            String::new(),
            String::new(),
            String::new(),
            String::new(),
        );

        let mut prev_layer_outputs = n_inputs as u32;

        // for backpropagate code generation
        let mut reversed_layers = self.structure.layers.to_vec();
        let mut next_layer_inputs = reversed_layers.pop().unwrap().n_neurons;
        reversed_layers = reversed_layers.into_iter().rev().collect::<Vec<Layer>>();
        reversed_layers.push(Layer {
            n_neurons: prev_layer_outputs,
            activation: Activation::Linear,
        });
        println!("{:?}\n{:?}", self.structure.layers, reversed_layers);

        let mut next_next_layer_inputs = 0;

        // purposefully not an iterator loop since the formatting gets ugly
        for i in 0..n_layers {
            let decrement = n_layers - i - 1;

            let n_neurons = self.structure.layers[i].n_neurons;
            let reverse_n_neurons = reversed_layers[i].n_neurons;

            i_weights += &format!("    weights{i}: array<array<f32, {prev_layer_outputs}>, {n_neurons}>,\n");
            i_biases += &format!("    biases{i}: array<f32, {n_neurons}>,\n");

            storage += &formatdoc! {"
                var<private> al{i}: array<f32, {n_neurons}>;
                var<private> delta{i}: array<f32, {n_neurons}>;
            "};

            let forward_input = if i != 0 {
                &format!("al{}", i - 1)
            } else { "X[id.x]" };
            
            let activation = if i != n_layers - 1 {
                &format!("al{i}[i] = relu(al{i}[i]);")
            } else { "" };

            // i did try to use the indent macro but- issues... 
            // and i still really wanna make it look pretty
            forward += &indent(&formatdoc! {"
                for (var i = 0; i < {n_neurons}; i++) {{
                    al{i}[i] = 0.0;
                    for (var j = 0; j < {prev_layer_outputs}; j++) {{
                        al{i}[i] += weights.weights{i}[i][j] * {forward_input}[j];
                    }}
                    al{i}[i] += biases.biases{i}[i];
                    {activation}
                }};

            "}, "    ");
                
            let backpropagation_input = if decrement != 0 {
                &format!{"al{}", decrement - 1}
            } else { "X[id.x]" };

            // the difference between the two is that if its the last layer of backprog, we calculate the delta
            // as just dal^L/dzl^L dC/dal^L whereas if not, its the sum((W^L)_i,j delta^L+1) (I THINK) 
            if i == 0 {
                backpropagate += &indent(&formatdoc! {"
                    for (var i = 0; i < {next_layer_inputs}; i++) {{
                        let tmp = (softmax_outputs[i] - targets[id.x][i]);
                        
                        for (var j = 0; j < {reverse_n_neurons}; j++) {{
                            outputs.grad_weights[id.x].weights{decrement}[i][j] = {backpropagation_input}[j] * tmp;
                            }}
                            
                        outputs.grad_biases[id.x].biases{decrement}[i] = tmp;

                        delta{decrement}[i] = tmp;
                    }}

                "}, "    ");
            } else {
                let next_layer = decrement + 1;
                backpropagate += &indent(&formatdoc! {"
                    for (var i = 0; i < {next_layer_inputs}; i++) {{
                        var sum = 0.0;
                        for (var j = 0; j < {next_next_layer_inputs}; j++) {{
                            sum += weights.weights{next_layer}[j][i] * delta{next_layer}[j];
                        }}

                        let tmp = sum * drelu(al{decrement}[i]);

                        for (var j = 0; j < {reverse_n_neurons}; j++) {{
                            outputs.grad_weights[id.x].weights{decrement}[i][j] = {backpropagation_input}[j] * tmp;
                        }}

                        outputs.grad_biases[id.x].biases{decrement}[i] = tmp;

                        delta{decrement}[i] = tmp;
                    }}

                "}, "    ");
            }

            prev_layer_outputs = n_neurons;
            next_next_layer_inputs = next_layer_inputs;
            next_layer_inputs = reverse_n_neurons;
        }

        template_wgsl(include_str!("neuralnet.wgsl").into(), &HashMap::from([
            ("batch_size".to_string(), self.structure.batch_size.to_string()),
            ("n_inputs".to_string(), n_inputs.to_string()),
            ("n_outputs".to_string(), prev_layer_outputs.to_string()),
            ("n_al".to_string(), (n_layers - 1).to_string()),
            ("i_weights".to_string(), i_weights),
            ("i_biases".to_string(), i_biases),
            ("storage".to_string(), storage),
            ("forward".to_string(), forward),
            ("backpropagate".to_string(), backpropagate),
        ])).into()
    }

    pub fn train(&mut self, options: &TrainingOptions) -> f32 {
        // flattening all the data initially so that we can get the bytelengths
        let current_batch: Vec<f32> = flatten2d(&self.batches[0]);
        let batch: &[u8] = bytemuck::cast_slice(&current_batch);

        let mut weights_v: Vec<f32> = flatten3d(&self.weights);
        let mut weights: &[u8] = bytemuck::cast_slice(&weights_v);
        let weights_bytelen = weights.len();

        let mut biases_v: Vec<f32> = flatten2d(&self.biases);
        let mut biases: &[u8] = bytemuck::cast_slice(&biases_v);
        let biases_bytelen = biases.len();

        let current_target: Vec<f32> = flatten2d(&self.targets[0]);
        let target: &[u8] = bytemuck::cast_slice(&current_target);

        let zeroed_outputs = vec![0.0; self.structure.batch_size + weights_v.len() + biases_v.len()];
        let outputs = bytemuck::cast_slice(&zeroed_outputs);

        // do i need an explanation for this :/
        // also currently not seperated into variables bcuz idk what to name said variables
        let outputs_indices = [
            self.structure.batch_size * size_of::<f32>(),
            self.structure.batch_size * (size_of::<f32>() + weights_bytelen),
            self.structure.batch_size * (size_of::<f32>() + weights_bytelen + biases_bytelen),
        ];
        let outputs_bytelen = (
            self.structure.batch_size
            * (size_of::<f32>() + weights_bytelen + biases_bytelen)
        ) as u64; // we could just also sum outputs_indices

        let nn_buffers = self.pipeline.create_buffers(
            batch.len() as u64, 
            weights_bytelen as u64,
            biases_bytelen as u64,
            target.len() as u64,
            outputs_bytelen,
        );

        let bind_group_layout = self.pipeline.create_bind_group_layout();

        let cs_pipeline_layout = self.pipeline.create_pipeline_layout(&bind_group_layout);
        let wgsl_code = self.generate_wgsl();
        let cs_module = self.pipeline.create_cs_module(wgsl_code);
        let cs_pipeline = self.pipeline.create_pipeline(&cs_pipeline_layout, &cs_module);

        let bind_group = self.pipeline.create_bind_group(&bind_group_layout, &nn_buffers);

        self.pipeline.queue.write_buffer(&nn_buffers.outputs_buf, 0, outputs); // no guarantees its zeros prior ;-;

        let mut cost: f32 = 0.0;
        for iteration in 0..options.iterations {
            let learning_rate = (options.learning_rate)(iteration);

            // not an iterator loop since self.batches and lifetimes iirc
            for i in 0..self.batches.len() - 2 {
                self.pipeline.queue.write_buffer(&nn_buffers.weights_buf, 0, weights);
                self.pipeline.queue.write_buffer(&nn_buffers.biases_buf, 0, biases);        
        
                let current_batch = flatten2d(&self.batches[i]);
                let batch: &[u8] = bytemuck::cast_slice(&current_batch);

                let current_target = flatten2d(&self.targets[i]);
                let target: &[u8] = bytemuck::cast_slice(&current_target);

                self.pipeline.queue.write_buffer(&nn_buffers.batch_buf, 0, batch);
                self.pipeline.queue.write_buffer(&nn_buffers.targets_buf, 0, target);

                let (tmp_cost, grad_weights, grad_biases) = self.pipeline.compute(
                    &cs_pipeline,
                    &bind_group,
                    &nn_buffers,
                    &outputs_indices,
                    outputs_bytelen,
                    self.structure.batch_size,
                ).block_on();
                println!("{:?}", tmp_cost);
                cost = tmp_cost;  // im honestly too lazy rn

                weights_v = weights_v.iter()
                    .enumerate()
                    .map(|(i, weight)| weight - learning_rate * grad_weights[i])
                    .collect::<Vec<f32>>();
                weights = bytemuck::cast_slice(&weights_v);

                biases_v = biases_v.iter()
                    .enumerate()
                    .map(|(i, bias)| bias - learning_rate * grad_biases[i])
                    .collect::<Vec<f32>>();
                biases = bytemuck::cast_slice(&biases_v);
            }

            println!("cost: {}, iteration: {}", cost, iteration);
        }

        let mut i = 0;
        for weight_matrix in self.weights.iter_mut() {
            for weight_vector in weight_matrix {
                for weight in weight_vector {
                    *weight = weights_v[i];
                    i += 1;
                }
            }
        }

        let mut i = 0;
        for bias_vector in self.biases.iter_mut() {
            for bias in bias_vector {
                *bias = biases_v[i];
                i += 1;
            }
        }

        return cost;
    }
}