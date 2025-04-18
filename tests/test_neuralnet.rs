use neuralnyx::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sin() {
        let mut inputs: Vec<Vec<f32>> = Vec::new();
        let mut outputs: Vec<Vec<f32>> = Vec::new();

        for i in 0..300 {
            inputs.push(vec![i as f32]);
            outputs.push(vec![(i as f32).sin()]);
        }

        let mut nn = NeuralNet::new(&mut inputs, &mut outputs, Structure {
            layers: vec![
                Layer {
                    neurons: 3,
                    activation: Activation::Relu,
                }, Layer {
                    neurons: 3,
                    activation: Activation::Relu,
                }, Layer {
                    neurons: 1,
                    activation: Activation::Sigmoid,
                }
            ],
            ..Default::default()
        }).unwrap();

        let cost = nn.train(&Default::default());
        println!("{cost}");
    }
}