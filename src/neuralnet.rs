use crate::pipeline;
use crate::structure;
use crate::utils;
use crate::types;

use core::fmt;
use rand::prelude::*;
use rand_distr::Normal;
use pollster::FutureExt;
use std::mem::size_of;

use structure::Structure;
use pipeline::NeuralNetPipeline;
use utils::{
    flatten2d,
    flatten3d,
    pad2d,
    shuffle_with_correspondence,
};
use types::{
    Layer,
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

impl fmt::Display for NeuralNet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "weights = {:?}\nbiases = {:?}", self.weights, self.biases)
    }
}

impl NeuralNet {
    // i mean- i guess inputs and outputs could be given during train() but
    // ill change it to that way later ig, ts is a mess
    pub fn new(
        inputs: &mut Vec<Vec<f32>>, 
        outputs: &mut Vec<Vec<f32>>, 
        structure: Structure,
    ) -> Result<NeuralNet, String> {
        // ~~~ checks to ensure variables are valid ~~~
        // putting it in a seperate function is ugly but ts ugly too, pmo
        // also i SWEAR this is js temporary </3
        if inputs.len() != outputs.len() {
            return Err("number of inputs must map to number of outputs"
                .to_string()
            );
        }

        let n_inputs: usize;
        if inputs.is_empty() {
            return Err("there's nothing to train on?...".to_string());
        } else {
            n_inputs = inputs[0].len();
            for input in inputs.iter() {
                if input.len() != n_inputs {
                    return Err(
                        "number of elements in input vectors must stay consistent"
                            .to_string()
                    );
                }
            }
        }
        
        let n_outputs: usize;
        if outputs.is_empty() {
            return Err(
                "uhh- i mean likee? i guess technically?... but u might as well just- coinflip"
                    .to_string()
            );
        } else {
            n_outputs = outputs[0].len();
            for output in outputs.iter() {
                if output.len() != n_outputs {
                    return Err(
                        "number of elements in output vectors must stay consistent"
                            .to_string()
                    );
                }
            }
        }

        structure.validate(n_outputs as u32)?;

        // the gpu pipeline
        let pipeline = NeuralNetPipeline::new().block_on();

        let (weights, biases) = NeuralNet::initialize(n_inputs, &structure.layers); // weight and biases initialization

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

    fn initialize(n_inputs: usize, layers: &Vec<Layer>) 
    -> (Vec<Vec<Vec<f32>>>, Vec<Vec<f32>>) {
        let mut rng = rand::rng();

        let mut weights: Vec<Vec<Vec<f32>>> = Vec::new();
        let mut biases: Vec<Vec<f32>> = Vec::new();
        
        let mut n_prev_outputs = n_inputs as u32;
        for layer in layers.iter() {
            // a weights matrix or a 2d vector 
            weights.push((0..layer.neurons)
                .map(|_| (0..n_prev_outputs)
                .map(|_| {
                    Normal::new(0.0, (2.0 / n_prev_outputs as f32).sqrt()) // kaiming initialization
                        .unwrap().sample(&mut rng) 
                }).collect()
            ).collect::<Vec<Vec<f32>>>());

            biases.push(vec![0.01; layer.neurons as usize]); // just so relu doesnt cause problems

            n_prev_outputs = layer.neurons;
        }

        (weights, biases)
    }

    pub fn train(&mut self, options: TrainingOptions) -> f32 {
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

        let zeroed_outputs = vec![
            0.0; self.structure.batch_size * (size_of::<f32>() + weights_v.len() + biases_v.len())
        ];
        let outputs = bytemuck::cast_slice(&zeroed_outputs);
        let outputs_bytelen = outputs.len();

        // do i need an explanation for this :/
        // also currently not seperated into variables bcuz idk what to name said variables
        let outputs_indices = [
            self.structure.batch_size * size_of::<f32>(),
            self.structure.batch_size * (size_of::<f32>() + weights_bytelen),
            self.structure.batch_size * (size_of::<f32>() + weights_bytelen + biases_bytelen),
        ];

        let nn_buffers = self.pipeline.create_buffers(
            batch.len() as u64,
            weights_bytelen as u64,
            biases_bytelen as u64,
            target.len() as u64,
            outputs_bytelen as u64,
        );

        let bind_group_layout = self.pipeline.create_bind_group_layout();

        let cs_pipeline_layout = self.pipeline.create_pipeline_layout(&bind_group_layout);
        
        let wgsl_code = self.structure.generate_wgsl(self.batches[0][0].len());
        let cs_module = self.pipeline.create_cs_module(wgsl_code);

        let cs_pipeline = self.pipeline.create_pipeline(&cs_pipeline_layout, &cs_module);

        let bind_group = self.pipeline.create_bind_group(&bind_group_layout, &nn_buffers);

        self.pipeline.queue.write_buffer(&nn_buffers.outputs_buf, 0, outputs); // no guarantees its zeros prior ;-;

        let n_batches = self.batches.len();

        // indices for the last batch
        let mut padded_index = n_batches - 1;
        let nonpadded_batch_size = self.batches[padded_index].len();
        let padded_batch_indices = [
            nonpadded_batch_size * size_of::<f32>(),
            outputs_indices[0] + nonpadded_batch_size * weights_bytelen,
            outputs_indices[1] +  nonpadded_batch_size * biases_bytelen,
        ];

        // pad the last batch because otherwise UB
        pad2d(&mut self.batches[padded_index], self.structure.batch_size);
        pad2d(&mut self.targets[padded_index], self.structure.batch_size);

        if options.verbose {
            println!("Training...");
        }

        let mut average_cost: f32 = 0.0;
        let mut t: usize = 1;

        // initialize the states of the optimizer
        let mut optimizer = options.optimizer.init(weights_v.len() + biases_v.len());

        for iteration in 0..options.epochs {
            average_cost = 0.0; // reset average cost after every epoch

            if options.shuffle_data {
                let shuffle_lookups = shuffle_with_correspondence(&mut self.batches, &mut self.targets);
                padded_index = shuffle_lookups[padded_index];
            }

            // not an iterator loop since self.batches and lifetimes iirc
            for i in 0..n_batches {
                self.pipeline.queue.write_buffer(&nn_buffers.weights_buf, 0, weights);
                self.pipeline.queue.write_buffer(&nn_buffers.biases_buf, 0, biases);        
        
                let current_batch = flatten2d(&self.batches[i]);
                let batch: &[u8] = bytemuck::cast_slice(&current_batch);

                let current_target = flatten2d(&self.targets[i]);
                let target: &[u8] = bytemuck::cast_slice(&current_target);

                self.pipeline.queue.write_buffer(&nn_buffers.batch_buf, 0, batch);
                self.pipeline.queue.write_buffer(&nn_buffers.targets_buf, 0, target);

                let mut current_batch_size = self.structure.batch_size;
                let mut batch_indices = outputs_indices;

                if i == padded_index {
                    current_batch_size = nonpadded_batch_size;
                    batch_indices = padded_batch_indices;
                }

                let (cost, grad_weights, grad_biases) = self.pipeline.compute(
                    &cs_pipeline,
                    &bind_group,
                    &nn_buffers,
                    &batch_indices,
                    &outputs_indices,
                    outputs_bytelen as u64,
                    current_batch_size,
                ).block_on();
                average_cost += cost;

                // update the weights and biases vectors
                let mut i = 0;
                for (j, weight) in weights_v.iter_mut().enumerate() {
                    *weight += optimizer.optimize(grad_weights[j], i, t);
                    i += 1;
                }
                weights = bytemuck::cast_slice(&weights_v);

                for (k, bias) in biases_v.iter_mut().enumerate() {
                    *bias += optimizer.optimize(grad_biases[k], i, t);
                    i += 1;
                }
                biases = bytemuck::cast_slice(&biases_v);
        
                t += 1;
            }

            average_cost /= n_batches as f32;
            if options.verbose {
                println!("Epoch: {} / {}, Cost: {}", iteration + 1, options.epochs, average_cost);
            }

            if options.cost_threshold > average_cost {
                break;
            }
        }

        // update the weights and biases from the flattened weights and biases
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

        return average_cost;
    }

    // below functions r js for testing duh, inefficient too
    pub fn test(&self, input: Vec<f32>) -> Vec<f32> {
        let mut output = input;
        
        for layer in 0..self.structure.layers.len() {
            let current_layer = self.structure.layers[layer];
            
            let neurons = current_layer.neurons as usize;
            let mut zl = vec![0.0; neurons];
            
            for i in 0..neurons {
                for j in 0..output.len() {
                    zl[i] += self.weights[layer][i][j] * output[j];
                }
                zl[i] += self.biases[layer][i];
            }

            current_layer.activation.activate(&mut zl);

            output = zl;
        }

        output
    }

    pub fn gauge(&self, inputs: &Vec<Vec<f32>>, targets: &Vec<Vec<f32>>) -> Result<f32, String> {
        let total = inputs.len();
        let mut correct = 0;
    
        for (input, target) in inputs.iter().zip(targets.iter()) {
            let predicted_label = self.test(input.clone());
    
            let mut max = 0.0;
            let mut predicted_max_index = 0;
            for (i, &val) in predicted_label.iter().enumerate() {
                if val > max {
                    max = val;
                    predicted_max_index = i;
                }
            }

            let mut max = 0.0;
            let mut true_max_index = 0;
            for (i, &val) in target.iter().enumerate() {
                if val > max {
                    max = val;
                    true_max_index = i;
                }
            }

            if predicted_max_index == true_max_index {
                correct += 1;
            }
        }

        Ok((correct as f32 / total as f32) * 100.0)
    }
}