use ndarray::{Array1, Array2, Axis, Ix1, LinalgScalar};
use num_traits::Zero;

use crate::traits::{Layer, Optimizer};

pub struct Add<V, O> {
    bias: Array1<V>,
    optimizer: O,
}

impl<V, O> Add<V, O> {
    pub fn new(bias: Array1<V>, optimizer: O) -> Self {
        Self { bias, optimizer }
    }

    pub fn new_zero(shape: usize, optimizer: O) -> Self
    where
        V: Clone + Zero,
    {
        Self::new(Array1::zeros(shape), optimizer)
    }
}

impl<V, O> Layer<Array2<V>, Array2<V>> for Add<V, O>
where
    V: LinalgScalar,
    O: Optimizer<V, Ix1>,
{
    fn forward(&self, input: &Array2<V>) -> Array2<V> {
        input + &self.bias
    }

    fn backward(&self, grad_out: &Array2<V>, _: &Array2<V>) -> Array2<V> {
        grad_out.to_owned()
    }

    fn learn(&mut self, grad_out: &Array2<V>, _: &Array2<V>) {
        self.optimizer
            .update(&mut self.bias, grad_out.sum_axis(Axis(0)))
    }
}
