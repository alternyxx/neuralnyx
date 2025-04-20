use neuralnyx::*;

#[cfg(test)]
mod tests {
    use super::*;

    /*
        It's worth noting this test will literally fail sometimes and that's just
        how neural networks are?... I mean like, most of the functions are pub(crate)
        so all I can test is the api... WHICH IS THE NEURAL NETWORK, and so even if
        i do write tests, it's just leaving it up to chance even if it's correct...
    */
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
            ..Default::default()
        });

        // percentage error
        assert_eq!(true, (nn.test(vec![1.57])[0] - 1.0).abs() / 1.0 < 0.05);
    }
}