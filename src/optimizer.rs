use crate::types;
use types::{Optimize, Optimizer};

struct SGD {
    lr: f32,
}

struct Momentum {
    lr: f32,
    m: f32,
    v: f32,
}

// AdaGrad requires every gradient to be stored, which isn't really feasible rn.
// struct AdaGrad {
//     lr: f32,
// }

struct RMSProp {
    lr: f32,
    v: f32,
    beta: f32,
    epsilon: f32,
}

struct Adam {
    lr: f32,
    m: f32,
    v: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    t: i32,
}

impl Optimize for SGD {
    fn optimize(&mut self, grad: f32) -> f32 {
        return -self.lr * grad;
    }
}

impl Optimize for Momentum {
    fn optimize(&mut self, grad: f32) -> f32 {
        self.v = self.m * self.v - self.lr * grad;

        return self.v;
    }
}

// impl Optimize for AdaGrad {
//     fn optimize(&mut self, grad: f32) -> f32 {
//         return self.lr * -grad;
//     }
// }

impl Optimize for RMSProp {
    fn optimize(&mut self, grad: f32) -> f32 {
        self.v = self.beta * self.v + (1.0 - self.beta) * grad.powi(2);

        return -self.lr * grad / self.epsilon + self.v.sqrt();
    }
}

impl Optimize for Adam {
    fn optimize(&mut self, grad: f32) -> f32 {
        self.t += 1;

        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad;
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * grad.powi(2);

        let beta1_t = self.beta1.powi(self.t);
        let beta2_t = self.beta2.powi(self.t);

        let mhat = self.m / (1.0 - beta1_t);
        let vhat = self.v / (1.0 - beta2_t);

        -self.lr * mhat / (self.epsilon + vhat.sqrt())
    }
}

impl Optimizer {
    // temp initialization
    pub(crate) fn init(&self) -> Box<dyn Optimize> {
        match self {
            Optimizer::SGD(lr) => Box::new(
                SGD{ lr: *lr }
            ),
            Optimizer::Momentum(lr, m) => Box::new(
                Momentum{
                    lr: *lr,
                    m: *m,
                    v: 0.0
                }
            ),
            // Optimizer::AdaGrad(lr) => Box::new(
            //     AdaGrad{ lr: *lr }
            // ),
            Optimizer::RMSProp(lr) => Box::new(
                RMSProp{
                    lr: *lr,
                    v: 0.0,
                    beta: 0.9,
                    epsilon: 1e-7,
                }
            ),
            Optimizer::Adam(lr) => Box::new(
                Adam{
                    lr: *lr,
                    m: 0.0,
                    v: 0.0,
                    beta1: 0.9,
                    beta2: 0.999,
                    epsilon: 1e-7,
                    t: 0,
                }
            ),
        }
    }
}