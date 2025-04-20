pub(crate) struct NeuralNetBuffers {
    pub batch_buf: wgpu::Buffer,
    pub weights_buf: wgpu::Buffer,
    pub biases_buf: wgpu::Buffer,
    pub targets_buf: wgpu::Buffer,
    pub outputs_buf: wgpu::Buffer,
    pub outputs_staging_buf: wgpu::Buffer,
}

pub(crate) struct NeuralNetPipeline {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl NeuralNetPipeline {
    // ik the pub(crate) is kinda redundant, but it looks prettier
    pub(crate) async fn new() -> Self {
        // env_logger::init(); // for debugging

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });
    
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: None,
            force_fallback_adapter: false,
        }).await.unwrap();
    
        let (device, queue) = adapter.request_device(
            &Default::default(),
            None,
        ).await.unwrap();
        
        Self {
            device,
            queue,
        }
    }

    pub(crate) fn create_buffers(
        &self,
        batch_len: u64,
        weights_len: u64,
        biases_len: u64,
        targets_len: u64,
        outputs_bytelen: u64,
    ) -> NeuralNetBuffers {
        // we should only need to check for the outputs_bytelen since thats the heaviest
        let limits = self.device.limits();
        if (outputs_bytelen > limits.max_buffer_size) || (outputs_bytelen as u32 > limits.max_storage_buffer_binding_size) {
            panic!(indoc::indoc! {"
                The specified archietecture exceeds limits.
                Try lowering the batch_size or the neurons in layers.
            "});
        }


        let batch_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("batch buffer"),
            size: batch_len,
            usage: 
                wgpu::BufferUsages::STORAGE 
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let weights_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("weights buffer"),
            size: weights_len,
            usage: 
                wgpu::BufferUsages::STORAGE 
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let biases_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("weights buffer"),
            size: biases_len,
            usage: 
                wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let targets_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("targets buffer"),
            size: targets_len,
            usage:
                wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let outputs_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("outputs buffer"),
            size: outputs_bytelen,
            usage: 
                wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let outputs_staging_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("outputs staging buffer"),
            size: outputs_bytelen,
            usage:
                wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        NeuralNetBuffers {
            batch_buf,
            weights_buf,
            biases_buf,
            targets_buf,
            outputs_buf,
            outputs_staging_buf,
        }
    }
    
    pub(crate) fn create_bind_group_layout(&self) -> wgpu::BindGroupLayout {
        self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { 
                            read_only: true,
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }, wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { 
                            read_only: true,
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }, wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { 
                            read_only: true,
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }, wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { 
                            read_only: true,
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }, wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                             read_only: false,
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    pub(crate) fn create_bind_group(
        &self,
        bind_group_layout: &wgpu::BindGroupLayout,
        nn_buffers: &NeuralNetBuffers,
    ) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bind group"),
            layout: bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: nn_buffers.batch_buf.as_entire_binding(),
                }, wgpu::BindGroupEntry {
                    binding: 1,
                    resource: nn_buffers.weights_buf.as_entire_binding(),
                }, wgpu::BindGroupEntry {
                    binding: 2,
                    resource: nn_buffers.biases_buf.as_entire_binding(),
                }, wgpu::BindGroupEntry {
                    binding: 3,
                    resource: nn_buffers.targets_buf.as_entire_binding(),
                }, wgpu::BindGroupEntry {
                    binding: 4,
                    resource: nn_buffers.outputs_buf.as_entire_binding(),
                },
            ]
        })
    }

    pub(crate) fn create_pipeline_layout(&self, bind_group_layout: &wgpu::BindGroupLayout)
    -> wgpu::PipelineLayout {
        self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("compute pipeline layout"),
            bind_group_layouts: &[bind_group_layout],
            push_constant_ranges: &[],
        })
    }

    pub(crate) fn create_cs_module(&self, wgsl_code: String) -> wgpu::ShaderModule {
        self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("forward propagation module"),
            source: wgpu::ShaderSource::Wgsl(wgsl_code.into()),
        })
    }

    pub(crate) fn create_pipeline(
        &self,
        cs_pipeline_layout: &wgpu::PipelineLayout,
        cs_module: &wgpu::ShaderModule,
    ) -> wgpu::ComputePipeline {
        self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("compute pipeline"),
            layout: Some(cs_pipeline_layout),
            module: cs_module,
            entry_point: Some("forward_pass"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        })
    }
    
    pub(crate) async fn compute(
        &self,
        cs_pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        nn_buffers: &NeuralNetBuffers,
        batch_indices: &[usize],    // this includes indices of the current batch's raw outputs
        outputs_indices: &[usize],  // whereas this is the true indices of the raw outputs
        outputs_bytelen: u64,
        batch_size: usize,
    ) -> (f32, Vec<f32>, Vec<f32>) {
        let mut encoder = self.device.create_command_encoder(&Default::default());

        // icl killing compute_pass instead of compute_pass.end() is so funny xD
        {
            let mut compute_pass = encoder.begin_compute_pass(&Default::default());

            compute_pass.set_pipeline(cs_pipeline);
            compute_pass.set_bind_group(0, bind_group, &[]);
            compute_pass.dispatch_workgroups(batch_size as u32, 1, 1);
        }

        encoder.copy_buffer_to_buffer(
            &nn_buffers.outputs_buf,
            0,
            &nn_buffers.outputs_staging_buf,
            0,
            outputs_bytelen,
        );

        self.queue.submit(Some(encoder.finish()));
    
        let outputs_buf_slice = nn_buffers.outputs_staging_buf.slice(..);

        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        outputs_buf_slice.map_async(wgpu::MapMode::Read, move |cost| {
            sender.send(cost).unwrap()
        });
    
        self.device.poll(wgpu::Maintain::Wait);

        if let Some(Ok(())) = receiver.receive().await {
            let average_cost: f32;
            let mut average_grad_weights: Vec<f32>;
            let mut average_grad_biases: Vec<f32>;

            {
                let outputs_raw = outputs_buf_slice.get_mapped_range();

                let costs: &[f32] = bytemuck::cast_slice(&outputs_raw[0..batch_indices[0]]);
                let grad_weights: &[f32] = bytemuck::cast_slice(
                    &outputs_raw[outputs_indices[0]..batch_indices[1]]
                );
                let grad_biases: &[f32] = bytemuck::cast_slice(
                    &outputs_raw[outputs_indices[1]..batch_indices[2]]
                );

                // averaging the costs over the batches
                let costs_sum: f32 = costs.iter().sum();
                average_cost = costs_sum / batch_size as f32; // batch_size == costs.len() here duh

                // average the weight gradients over the batches (with worst performance)
                let batches_grad_weights = grad_weights
                    .chunks(grad_weights.len() / batch_size)
                    .map(|v| v.to_vec())
                    .collect::<Vec<Vec<f32>>>();

                let weights_len = batches_grad_weights[0].len();
                average_grad_weights = vec![0.0; weights_len];

                for i in 0..weights_len {
                    for batch in 0..batch_size {
                        average_grad_weights[i] += batches_grad_weights[batch][i];
                    }
                }
                for grad_weight in average_grad_weights.iter_mut() {
                    *grad_weight /= batch_size as f32;
                }

                // average the biases gradients over the batches (with worst performance)
                let batches_grad_biases = grad_biases
                    .chunks(grad_biases.len() / batch_size)
                    .map(|v| v.to_vec())
                    .collect::<Vec<Vec<f32>>>();

                let biases_len = batches_grad_biases[0].len();
                average_grad_biases = vec![0.0; biases_len];

                for i in 0..biases_len {
                    for batch in 0..batch_size {
                        average_grad_biases[i] += batches_grad_biases[batch][i];
                    }
                }

                for grad_bias in average_grad_biases.iter_mut() {
                    *grad_bias /= batch_size as f32;
                }
            }

            nn_buffers.outputs_staging_buf.unmap();
            
            (average_cost, average_grad_weights, average_grad_biases)
        } else {
            panic!("uhm");
        }
    }
}