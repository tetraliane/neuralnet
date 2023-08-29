use ndarray::{Array2, Ix2, ScalarOperand};
use num_traits::Float;

use crate::traits::Terminal;
use crate::utils::cross_entropy_error;

pub struct CrossEntropy {}

impl CrossEntropy {
    pub fn new() -> Self {
        Self {}
    }
}

impl<V> Terminal<V, Ix2> for CrossEntropy
where
    V: Float + ScalarOperand,
{
    fn predict(&self, input: &Array2<V>) -> Array2<V> {
        input.to_owned()
    }

    fn loss(&self, input: &Array2<V>, teacher: &Array2<V>) -> V {
        cross_entropy_error(input, teacher)
    }

    fn fit(&mut self, input: &Array2<V>, teacher: &Array2<V>) -> Array2<V> {
        let batch_size =
            V::from(teacher.dim().0).expect("Failed to cast usize into the value type");
        -teacher.to_owned() / input / batch_size
    }
}
