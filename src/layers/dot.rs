use ndarray::{Array2, Ix2, LinalgScalar};

use crate::traits::{Layer, Optimizer};

pub struct Dot<V, O> {
    wgt: Array2<V>,
    optimizer: O,
}

impl<V, O> Dot<V, O> {
    pub fn new(wgt: Array2<V>, optimizer: O) -> Self {
        Self { wgt, optimizer }
    }
}

impl<V, O> Layer<Array2<V>, Array2<V>> for Dot<V, O>
where
    V: LinalgScalar,
    O: Optimizer<V, Ix2>,
{
    fn forward(&self, input: &Array2<V>) -> Array2<V> {
        input.dot(&self.wgt)
    }

    fn backward(&self, grad_out: &Array2<V>, _: &Array2<V>) -> Array2<V> {
        grad_out.dot(&self.wgt.t())
    }

    fn learn(&mut self, grad_out: &Array2<V>, input: &Array2<V>) {
        self.optimizer
            .update(&mut self.wgt, input.t().dot(grad_out))
    }
}
