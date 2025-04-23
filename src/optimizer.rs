use crate::types;
use types::{Optimize, Optimizer};

/*
    I'll probably update this implementation in some time.
    Since rn, this cu
*/

pub struct SGD {
    lr: f32,
}

pub struct Momentum {
    lr: f32,
    m: f32,
    v: f32,
}

// struct AdaGrad {
//     lr: f32,
//     epsilion: f32,
// }

pub struct RMSProp {
    lr: f32,
    v: Vec<f32>,
    beta: f32,
    epsilon: f32,
}

pub struct Adam {
    lr: f32,
    m: Vec<f32>,
    v: Vec<f32>,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
}

impl Optimize for SGD {
    fn optimize(&mut self, grad: f32, _: usize, _: usize) -> f32 {
        -self.lr * grad
    }
}

impl Optimize for Momentum {
    fn optimize(&mut self, grad: f32, _: usize, _: usize) -> f32 {
        self.v = self.m * self.v - self.lr * grad;

        return self.v;
    }
}

// impl Optimize for AdaGrad {
//     fn optimize(&mut self, grad: f32) -> f32 {
        

//         -self.lr * grad / grad
//     }
// }

// index lookups can be minimzed sure but this is cleaner :P
impl Optimize for RMSProp {
    fn optimize(&mut self, grad: f32, i: usize, _: usize) -> f32 {
        self.v[i] = self.beta * self.v[i] + (1.0 - self.beta) * grad.powi(2);

        -self.lr * grad / (self.epsilon + self.v[i].sqrt())
    }
}

impl Optimize for Adam {
    fn optimize(&mut self, grad: f32, i: usize, t: usize) -> f32 {
        self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grad;
        self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * grad.powi(2);

        let beta1_t = self.beta1.powi(t as i32);    // yikes
        let beta2_t = self.beta2.powi(t as i32);

        let mhat = self.m[i] / (1.0 - beta1_t);
        let vhat = self.v[i] / (1.0 - beta2_t);

        -self.lr * mhat / (self.epsilon + vhat.sqrt())
    }
}

impl Optimizer {
    // temp initialization
    pub(crate) fn init(self, n_params: usize) -> Box<dyn Optimize> {
        match self {
            Optimizer::SGD(lr) => Box::new(
                SGD{ lr }
            ),
            Optimizer::Momentum(lr, m) => Box::new(
                Momentum{
                    lr,
                    m: m,
                    v: 0.0
                }
            ),
            // Optimizer::AdaGrad(lr) => Box::new(
            //     AdaGrad{ lr: *lr }
            // ),
            Optimizer::RMSProp(lr) => Box::new(
                RMSProp{
                    lr,
                    v: vec![0.0; n_params],
                    beta: 0.9,
                    epsilon: 1e-7,
                }
            ),
            Optimizer::Adam(lr) => Box::new(
                Adam{
                    lr,
                    m: vec![0.0; n_params],
                    v: vec![0.0; n_params],
                    beta1: 0.9,
                    beta2: 0.999,
                    epsilon: 1e-8,
                }
            ),
            Optimizer::Custom(optimizer) => optimizer,
        }
    }
}