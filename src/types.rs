use core::fmt;

// #[cfg(feature = "use-f16")]
// enum Float {

// }

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Activation {
    Linear,
    Tanh,
    Relu,
    Sigmoid,
    Softmax,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum CostFunction {
    MeanSquaredError,   // idk if i should acronomize this to mse?..
    CrossEntropy,
}

#[derive(Copy, Clone, Debug)]
pub struct Layer {
    pub neurons: u32,
    pub activation: Activation,
}

pub trait Optimize{
    fn optimize(&mut self, grad: f32, i: usize, t: usize) -> f32;
}

pub enum Optimizer {
    SGD(f32),
    Momentum(f32, f32),
    RMSProp(f32),
    Adam(f32),
    // this is currently p useless since n_parameters cant be accessed but
    // just here for later implementations
    Custom(Box<dyn Optimize>),
}

pub struct TrainingOptions {
    pub optimizer: Optimizer, // this is to allow for decreasing learning rates and such
    pub epochs: u32,
    pub cost_threshold: f32,
    pub shuffle_data: bool,
    pub verbose: bool,
}

impl fmt::Display for Activation {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Activation::Linear => write!(f, "linear"),
            Activation::Tanh => write!(f, "tanh"),
            Activation::Relu => write!(f, "relu"),
            Activation::Sigmoid => write!(f, "sigmoid"),
            Activation::Softmax => write!(f, "softmax_activation"), // ts pmo icl but idk
        }
    }
}

impl fmt::Display for CostFunction {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            CostFunction::MeanSquaredError => write!(f, "mean_squared_error"),
            CostFunction::CrossEntropy => write!(f, "categorial_cross_entropy"),
        }
    }
}

impl fmt::Display for Layer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Layer {{ neurons: {}, activation: {} }}", self.neurons, self.activation)
    }
}

impl Default for TrainingOptions {
    fn default() -> Self {
        Self {
            optimizer: Optimizer::Adam(0.001),
            epochs: 3000,
            shuffle_data: false,
            cost_threshold: 0.01,
            verbose: false,
        }
    }
}