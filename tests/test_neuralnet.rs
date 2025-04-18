use neuralnyx::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sin() {
        let mut x: Vec<Vec<f32>> = Vec::new();
        let mut y: Vec<Vec<f32>> = Vec::new();
    
        let mut i = 0.0;
    
        while i < 6.15 {
            x.push(vec![i]);
            y.push(vec![i.sin()]);
        
            i += 0.01;
        }
    
        let layers = vec![
            Layer {
                neurons: 100,
                activation: Activation::Tanh,
            }, Layer {
                neurons: 1,
                activation: Activation::Linear,
            }
        ];
    
        let mut nn = NeuralNet::new(&mut x, &mut y, Structure {
            layers,
            ..Default::default()    
        }).unwrap();
    
        nn.train(&TrainingOptions {
            optimizer: Optimizer::Adam(0.001),
            epochs: 3000,
            verbose: false,
        });

        // percentage error
        assert_eq!(true, (nn.test(vec![1.57])[0] - 1.0).abs() / 1.0 < 0.01);
    }
}