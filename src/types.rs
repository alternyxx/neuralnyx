use core::fmt;

#[derive(Copy, Clone, Debug)]
pub enum Activation {
    Linear,
    Tanh,
    Relu,
    Sigmoid,
    Softmax,
}

#[derive(Copy, Clone)]
pub enum CostFunction {
    MeanSquaredError,
    CrossEntropy,
}

#[derive(Copy, Clone, Debug)]
pub struct Layer {
    pub n_neurons: u32,
    pub activation: Activation,
}

pub struct Structure {
    pub batch_size: usize,   // the amount of batches sent to the gpu per compute
    pub layers: &'static [Layer],  // layers of the neuralnet
    pub cost_function: CostFunction,  // cost function, available are mean_squared_error and cross entropy
}

pub struct TrainingOptions {
    pub learning_rate: fn(_: u32) -> f32, // this is to allow for decreasing learning rates and such
    pub iterations: u32,
}

impl fmt::Display for Activation {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Activation::Linear => write!(f, "linear"),
            Activation::Tanh => write!(f, "tanh"),
            Activation::Relu => write!(f, "relu"),
            Activation::Sigmoid => write!(f, "sigmoid"),
            Activation::Softmax => write!(f, "softmax_activation"),
        }
    }
}

impl fmt::Display for Layer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Layer {{ n_neurons: {}, activation: {} }}", self.n_neurons, self.activation)
    }
}

impl Default for Structure {
    fn default() -> Self {
        Self {
            batch_size: 64,
            layers: &[],
            cost_function: CostFunction::MeanSquaredError,
        }
    }
}

impl Default for TrainingOptions {
    fn default() -> Self {
        Self {
            learning_rate: |_| 0.01,
            iterations: 2000,
        }
    }
}