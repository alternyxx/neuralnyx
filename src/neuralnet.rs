use rand::prelude::*;
use rand_distr::Normal;
use indoc::formatdoc;
use textwrap::indent;
use pollster::FutureExt;
use std::collections::HashMap;
use std::mem::size_of;
use pipeline::NeuralNetPipeline;
use utils::{flatten3d, flatten2d, template_wgsl};

use crate::pipeline;
use crate::utils;

pub struct NeuralNet {
    pipeline: NeuralNetPipeline,    // the gpu pipeline and its functions
    batches: Vec<Vec<Vec<f32>>>,          // for these two, the inner vec is the vector inputs, or vec9f,
    targets: Vec<Vec<Vec<f32>>>, // for middle vec, its a single batch, and the outer vec groups the batches
    layers: Vec<i32>,   // vec.length() is the number of layers, i32 is the amount of neurons
    weights: Vec<Vec<Vec<f32>>>,    // for this, we can consider the outer vec as the layers and the two inner as a matrix
    biases: Vec<Vec<f32>>,  // outer vec is layers, inner vec is the vector of biases :/
    batch_size: usize,
}

pub struct NeuralNetOptions {
    batch_size: usize,
}

pub struct TrainingOptions {
    learning_rate: fn(_: u32) -> f32, // this is to allow for decreasing learning rates and such
    iterations: u32,
}

impl Default for NeuralNetOptions {
    fn default() -> Self {
        Self {
            batch_size: 64,
        }
    }
}

impl Default for TrainingOptions {
    fn default() -> Self {
        Self {
            learning_rate: |_| 0.01,
            iterations: 10000,
        }
    }
}

impl NeuralNet {
    // more direct approach to create NeuralNet
    pub fn new(
        inputs: &mut Vec<Vec<f32>>, 
        outputs: &mut Vec<Vec<f32>>, 
        layers: &[i32], 
        options: &NeuralNetOptions,
    ) -> Result<NeuralNet, String> {
        // ~~~ checks to ensure variables are valid ~~~     
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

        let mut layers = layers.to_vec();
        layers.push(n_output_v as i32); // internally, we want to have the last layer be the same as n_output_v, trust
        let (weights, biases) = NeuralNet::initialize(n_input_v, &layers); // weight and biases initialization

        // ~~~ seperate the inputs into batches ~~~
        let batches: Vec<Vec<Vec<f32>>> = inputs
            .chunks(options.batch_size)
            .map(|s| s.to_vec())
            .collect();
        let targets: Vec<Vec<Vec<f32>>> = outputs
            .chunks(options.batch_size)
            .map(|s| s.to_vec())
            .collect();

        Ok(Self {
            pipeline,
            batches,
            targets,
            layers,
            weights,
            biases,
            batch_size: options.batch_size as usize,
        })
    }

    fn initialize(n_input_v: usize, layers: &Vec<i32>) 
    -> (Vec<Vec<Vec<f32>>>, Vec<Vec<f32>>) {
        let mut rng = rand::rng();

        let mut weights: Vec<Vec<Vec<f32>>> = Vec::new();
        let mut biases: Vec<Vec<f32>> = Vec::new();
        
        let mut n_prev_outputs = n_input_v as i32;
        for n_neurons in layers.iter() {
            weights.push((0..*n_neurons)
                .map(|_| (0..n_prev_outputs)
                .map(|_| {
                    Normal::new(0.0, (2.0 / n_prev_outputs as f32).sqrt())
                        .unwrap().sample(&mut rng)
                }).collect()
            ).collect::<Vec<Vec<f32>>>());

            biases.push(vec![0.01; *n_neurons as usize]);

            n_prev_outputs = *n_neurons;
        }
    
        (weights, biases)
    }

    fn generate_wgsl(&self) -> String {
        // dynamic code generation
        let n_layers = self.layers.len();
        let n_inputs = self.batches[0][0].len();
        let mut i_weights = String::new();
        let mut i_biases = String::new();
        let mut forward = String::new();

        let mut x_parameter = "X[id.x]".to_string();
        let mut prev_outputs = n_inputs as i32;
        for (i, n_neurons) in self.layers.iter().enumerate() {
            i_weights += &format!("    weights{i}: array<array<f32, {prev_outputs}>, {n_neurons}>,\n");
            i_biases += &format!("    biases{i}: array<f32, {n_neurons}>,\n");
            
            let mut relu = String::new();
            if i != n_layers - 1 {
                relu += &format!("al{i}[i] = relu(al{i}[i]);");
            }

            // i did try to use the indent macro but- issues... 
            // and i still wanna make it look pretty
            forward += &indent(&formatdoc! {"
                var al{i} = array<f32, {n_neurons}>();
                for (var i = 0; i < {n_neurons}; i += 1) {{
                    for (var j = 0; j < {prev_outputs}; j += 1) {{
                        al{i}[i] += weights.weights{i}[i][j] * {x_parameter}[j];
                    }}
                    al{i}[i] += biases.biases{i}[i];
                    {relu}
                }};

            "}, "    ");

            x_parameter = format!("al{i}");
            prev_outputs = *n_neurons;
        }
            
        template_wgsl(include_str!("neuralnet.wgsl").into(), HashMap::from([
            ("batch_size".to_string(), self.batch_size.to_string()),
            ("n_inputs".to_string(), n_inputs.to_string()),
            ("n_outputs".to_string(), self.layers[n_layers - 1].to_string()),
            ("n_al".to_string(), (n_layers - 1).to_string()),
            ("i_weights".to_string(), i_weights),
            ("i_biases".to_string(), i_biases),
            ("forward".to_string(), forward),
        ])).into()
    }

    pub fn train(&self, options: &TrainingOptions) {
        // flattening all the data initially so that we can get the bytelengths
        let current_batch: Vec<f32> = flatten2d(&self.batches[0]);
        let batch: &[u8] = bytemuck::cast_slice(&current_batch);

        let mut weights_v: Vec<f32> = flatten3d(&self.weights);
        let weights: &[u8] = bytemuck::cast_slice(&weights_v);
        let weights_bytelen = weights.len();
        
        let mut biases_v: Vec<f32> = flatten2d(&self.biases);
        let biases: &[u8] = bytemuck::cast_slice(&biases_v);
        let biases_bytelen = biases.len();

        let current_target: Vec<f32> = flatten2d(&self.targets[0]);
        let target: &[u8] = bytemuck::cast_slice(&current_target);

        let zeroed_outputs = vec![0.0; self.batch_size + weights_v.len() + biases_v.len()];
        let outputs = bytemuck::cast_slice(&zeroed_outputs);

        // do i need an explanation for this :/
        // also currently not seperated into variables bcuz idk what to name said variables
        let outputs_indices = [
            self.batch_size * size_of::<f32>(),
            self.batch_size * (size_of::<f32>() + weights_bytelen),
            self.batch_size * (size_of::<f32>() + weights_bytelen + biases_bytelen),
        ];
        let outputs_bytelen = (
            self.batch_size
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

        self.pipeline.queue.write_buffer(&nn_buffers.weights_buf, 0, weights);
        self.pipeline.queue.write_buffer(&nn_buffers.biases_buf, 0, biases);
        self.pipeline.queue.write_buffer(&nn_buffers.outputs_buf, 0, outputs); // even if we're js zeroing, no guarantees its zeros prior


        for iteration in 0..options.iterations {
            let learning_rate = (options.learning_rate)(iteration);
            let mut cost: f32 = 0.0;

            // not an iterator loop since self.batches and lifetimes iirc
            for i in 0..self.batches.len() - 2 {
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
                    self.batch_size,
                ).block_on();
                cost = tmp_cost;
                
                weights_v = weights_v.iter()
                    .enumerate()
                    .map(|(i, weight)| weight - learning_rate * grad_weights[i])
                    .collect::<Vec<f32>>();
                let weights = bytemuck::cast_slice(&weights_v);

                biases_v = biases_v.iter()
                    .enumerate()
                    .map(|(i, bias)| bias - learning_rate * grad_biases[i])
                    .collect::<Vec<f32>>();
                let biases = bytemuck::cast_slice(&biases_v);

                self.pipeline.queue.write_buffer(&nn_buffers.weights_buf, 0, weights);
                self.pipeline.queue.write_buffer(&nn_buffers.biases_buf, 0, biases);
            }
            
            println!("cost: {}, iteration: {}", cost, iteration);
        }
    
        println!("{:?}", weights_v)
    }
}
