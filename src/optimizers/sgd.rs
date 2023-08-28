use std::ops::{Mul, SubAssign};

use ndarray::{Array, Dimension};

use crate::traits::Optimizer;

pub struct Sgd<V> {
    learning_rate: V,
}

impl<V> Sgd<V> {
    pub fn new(learning_rate: V) -> Self {
        Self { learning_rate }
    }
}

impl<V, D> Optimizer<V, D> for Sgd<V>
where
    D: Dimension,
    V: Mul<Array<V, D>, Output = Array<V, D>> + SubAssign + Clone,
{
    fn update(&mut self, param: &mut Array<V, D>, grad: Array<V, D>) {
        *param -= &(self.learning_rate.clone() * grad);
    }
}
