use ndarray::{Array1, Array2, Axis, ScalarOperand};
use num_traits::Float;

use crate::traits::Layer;
use crate::utils::softmax;

pub struct Softmax {}

impl Softmax {
    pub fn new() -> Self {
        Self {}
    }
}

impl<V: Float + ScalarOperand> Layer<Array2<V>> for Softmax {
    type Output = Array2<V>;

    fn forward(&self, input: &Array2<V>) -> Array2<V> {
        softmax(input)
    }

    fn backward(&self, grad_out: &Array2<V>, input: &Array2<V>) -> Array2<V> {
        let output = self.forward(input);
        let dot = output
            .axis_iter(Axis(0))
            .zip(grad_out.axis_iter(Axis(0)))
            .map(|(out, grad)| out.dot(&grad))
            .collect::<Array1<_>>()
            .into_shape((output.dim().0, 1))
            .expect("Failed to collect values into an array");
        output * (grad_out - &dot)
    }

    fn learn(&mut self, _: &Array2<V>, _: &Array2<V>) {}
}
