use core::fmt;

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

pub enum Optimizer {
    SGD(f32),
    Momentum(f32, f32),
    // AdaGrad(f32),
    RMSProp(f32),
    Adam(f32),
}

pub struct TrainingOptions {
    pub optimizer: Optimizer, // this is to allow for decreasing learning rates and such
    pub epochs: u32,
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
            epochs: 500,
            verbose: false,
        }
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
                
                // calculate e_i^zl
                let mut sum = 1.0e-20;
                for i in 0..n_logits {
                    let tmp = (zl[i] - highest).exp();
                    zl[i] = tmp;
                    sum += tmp; 
                }
            
                // e_i^zl / sum(e^zl)
                for i in 0..n_logits {
                    zl[i] /= sum;
                }
            },
        }
    }
}