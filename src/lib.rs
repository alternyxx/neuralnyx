#![doc = include_str!("../README.md")]

mod types;
mod utils;
mod pipeline;
mod structure;
mod optimizer;
mod neuralnet;

pub use self::types::Activation;
pub use self::types::CostFunction;
pub use self::types::Optimizer;
pub use self::types::Layer;
pub use self::types::TrainingOptions;

pub use self::structure::Structure;

pub use self::neuralnet::NeuralNet;