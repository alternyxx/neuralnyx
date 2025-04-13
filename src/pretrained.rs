use neuralnet::NeuralNet;
use types::Layer;

use crate::neuralnet;
use crate::types;

pub struct PreTrained {
    pub weights: Vec<Vec<Vec<f32>>>,
    pub biases: Vec<Vec<f32>>,
    pub layers: &'static [Layer]
}

impl From<NeuralNet> for PreTrained {
    fn from(nn: NeuralNet) -> Self {
        Self {
            weights: nn.weights,
            biases: nn.biases,
            layers: nn.structure.layers,
        }
    }
}