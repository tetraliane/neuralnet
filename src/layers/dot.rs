use ndarray::{Array2, Ix2, LinalgScalar, ScalarOperand};
use num_traits::Float;
use rand::{distributions::Standard, prelude::Distribution, Rng};

use crate::traits::{Layer, Optimizer};

pub struct Dot<V, O> {
    wgt: Array2<V>,
    optimizer: O,
}

impl<V, O> Dot<V, O> {
    pub fn new(wgt: Array2<V>, optimizer: O) -> Self {
        Self { wgt, optimizer }
    }

    pub fn new_random<R>(shape: (usize, usize), optimizer: O, rng: &mut R, std: V) -> Self
    where
        V: Float + ScalarOperand,
        Standard: Distribution<V>,
        R: Rng,
    {
        Self::new(random_array(shape, rng) * std, optimizer)
    }
}

impl<V, O> Layer<V, Ix2> for Dot<V, O>
where
    V: LinalgScalar,
    O: Optimizer<V, Ix2>,
{
    fn params(&self) -> Option<ndarray::ArrayViewD<V>> {
        Some(self.wgt.view().into_dyn())
    }

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

pub(crate) fn random_array<V, R>(shape: (usize, usize), rng: &mut R) -> Array2<V>
where
    V: Float + ScalarOperand,
    Standard: Distribution<V>,
    R: Rng,
{
    Array2::from_shape_vec(
        shape,
        rng.sample_iter(Standard).take(shape.0 * shape.1).collect(),
    )
    .expect("Failed to make the random array")
}
