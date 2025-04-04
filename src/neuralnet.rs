use rand::prelude::*;
use rand_distr::Normal;
use indoc::formatdoc;
use textwrap::indent;
use pollster::FutureExt;
use std::collections::HashMap;
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
    batch_size: u32,
}

pub struct NeuralNetOptions {
    batch_size: u32,
}

impl Default for NeuralNetOptions {
    fn default() -> Self {
        Self {
            batch_size: 128,
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
        let batches: Vec<Vec<Vec<f32>>> = inputs.chunks(options.batch_size as usize).map(|s| s.into()).collect();
        let targets: Vec<Vec<Vec<f32>>> = outputs.chunks(options.batch_size as usize).map(|s| s.into()).collect();

        Ok(Self {
            pipeline,
            batches,
            targets,
            layers,
            weights,
            biases,
            batch_size: options.batch_size,
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

    pub fn train(&mut self, _learning_rate: f32) {
        // flattening all the data initially so that we can get the bytelengths
        let mut current_batch: Vec<f32> = flatten2d(&self.batches[0]);
        let batch: &[u8] = bytemuck::cast_slice(&current_batch);

        let mut weights_v: Vec<f32> = flatten3d(&self.weights);
        let weights: &[u8] = bytemuck::cast_slice(&weights_v);

        let mut biases_v: Vec<f32> = flatten2d(&self.biases);
        let biases: &[u8] = bytemuck::cast_slice(&biases_v);

        let current_target: Vec<f32> = flatten2d(&self.targets[0]);
        let target: &[u8] = bytemuck::cast_slice(&current_target);

        let costs_v: Vec<f32> = vec![0.0; self.batch_size as usize];
        let costs: &[u8] = bytemuck::cast_slice(&costs_v);
        let costs_len = costs.len() as u64;

        let nn_buffers = self.pipeline.create_buffers(
            batch.len() as u64, 
            weights.len() as u64,
            biases.len() as u64,
            target.len() as u64,
            costs_len,
        );

        let bind_group_layout = self.pipeline.create_bind_group_layout();

        let cs_pipeline_layout = self.pipeline.create_pipeline_layout(&bind_group_layout);
        let wgsl_code = self.generate_wgsl();
        let cs_module = self.pipeline.create_cs_module(wgsl_code);
        let cs_pipeline = self.pipeline.create_pipeline(&cs_pipeline_layout, &cs_module);

        let bind_group = self.pipeline.create_bind_group(&bind_group_layout, &nn_buffers);

        self.pipeline.queue.write_buffer(&nn_buffers.batch_buf, 0, batch);
        self.pipeline.queue.write_buffer(&nn_buffers.targets_buf, 0, target);
        self.pipeline.queue.write_buffer(&nn_buffers.costs_buf, 0, costs);

        let mut rng = rand::rng();

        let mut best_average_cost: f32 = 20.0;
        let mut best_weights = weights_v.clone();
        let mut best_biases = biases_v.clone();

        for a in 0..100000 {
            weights_v = weights_v.iter().map(|w| w + rng.random_range(-3.00..3.00)).collect::<Vec<f32>>();
            let weights = bytemuck::cast_slice(&weights_v);
            biases_v = biases_v.iter().map(|b| b + rng.random_range(-3.00..3.00)).collect::<Vec<f32>>();
            let biases = bytemuck::cast_slice(&biases_v);
            
            self.pipeline.queue.write_buffer(&nn_buffers.weights_buf, 0, weights);
            self.pipeline.queue.write_buffer(&nn_buffers.biases_buf, 0, biases);
            
            // this is done this way because the variables are previously required for bytelength
            let mut average_cost = self
                .compute(&cs_pipeline, &bind_group, &nn_buffers.costs_buf, &nn_buffers.costs_staging_buf, &costs_len)
                .block_on();
        
            for i in 1..self.batches.len() - 2 {
                current_batch = self.batches[i].iter().flatten().copied().collect();
                let batch: &[u8] = bytemuck::cast_slice(&current_batch);
                
                self.pipeline.queue.write_buffer(&nn_buffers.batch_buf, 0, batch);
                    
                average_cost += self
                    .compute(&cs_pipeline, &bind_group, &nn_buffers.costs_buf, &nn_buffers.costs_staging_buf, &costs_len)
                    .block_on();
            }

            average_cost /= (self.batches.len() - 1) as f32;
            if average_cost < best_average_cost {
                best_weights = weights_v.clone();
                best_biases = biases_v.clone();
                best_average_cost = average_cost;
                println!("cost: {}, iteration: {a}", best_average_cost);
            } else {
                weights_v = best_weights.clone();
                biases_v = best_biases.clone();
            }
        }
    }
    
    async fn compute(
        &mut self, 
        cs_pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        costs_buf: &wgpu::Buffer,
        costs_staging_buf: &wgpu::Buffer,
        costs_len: &u64,
    ) -> f32 {
        let mut encoder = self.pipeline.device.create_command_encoder(&Default::default());

        // icl killing compute_pass instead of compute_pass.end() is so funny xD
        {
            let mut compute_pass = encoder.begin_compute_pass(&Default::default());
    
            compute_pass.set_pipeline(cs_pipeline);
            compute_pass.set_bind_group(0, bind_group, &[]);
            compute_pass.dispatch_workgroups(self.batch_size, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&costs_buf, 0, &costs_staging_buf, 0, *costs_len);

        self.pipeline.queue.submit(Some(encoder.finish()));
    
        let costs_buf_slice = costs_staging_buf.slice(..);

        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        costs_buf_slice.map_async(wgpu::MapMode::Read, move |cost| {
            sender.send(cost).unwrap()
        });
    
        self.pipeline.device.poll(wgpu::Maintain::Wait);
    
        // like srsly- i have to copy this from compute shaders 101
        let average_cost: f32;
        if let Some(Ok(())) = receiver.receive().await {
            // killing the costs_raw so that we can unmap the buffer
            {
                let costs_raw = &*costs_buf_slice.get_mapped_range();
                let costs: &[f32] = bytemuck::cast_slice(costs_raw);
                let costs_sum: f32 = costs.iter().sum();
                average_cost = costs_sum / costs.len() as f32; 
            }

            costs_staging_buf.unmap();
        } else {
            panic!("uhm");
        }

        average_cost
    }
}
