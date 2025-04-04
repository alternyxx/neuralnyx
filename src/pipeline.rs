pub(crate) struct NeuralNetBuffers {
    pub batch_buf: wgpu::Buffer,
    pub weights_buf: wgpu::Buffer,
    pub biases_buf: wgpu::Buffer,
    pub targets_buf: wgpu::Buffer,
    pub costs_buf: wgpu::Buffer,
    pub costs_staging_buf: wgpu::Buffer,
}

pub(crate) struct NeuralNetPipeline {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl NeuralNetPipeline {
    // ik the pub(crate) is kinda redundant, but it looks prettier
    pub(crate) async fn new() -> Self {
        env_logger::init(); // for debugging

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
        costs_len: u64,
    ) -> NeuralNetBuffers {
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
            size: weights_len as u64,
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

        let costs_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cost buffer"),
            size: costs_len,
            usage: 
                wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let costs_staging_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cost staging buffer one"),
            size: costs_len,
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
            costs_buf,
            costs_staging_buf,
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
                            read_only: true 
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
                            read_only: true 
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
                            read_only: true 
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
                            read_only: true 
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
                             read_only: false 
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
                    resource: nn_buffers.costs_buf.as_entire_binding(),
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
}