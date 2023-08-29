use ndarray::{Array2, Ix2, ScalarOperand};
use num_traits::Float;

use crate::traits::Terminal;
use crate::utils::{cross_entropy_error, softmax};

pub struct SoftmaxWithLoss {}

impl SoftmaxWithLoss {
    pub fn new() -> Self {
        Self {}
    }

    fn loss<V>(&self, input: &Array2<V>, teacher: &Array2<V>) -> (Array2<V>, V)
    where
        V: Float + ScalarOperand,
    {
        let y = softmax(input);
        let error = cross_entropy_error(&y, teacher);
        (y, error)
    }

    fn backward<V>(&self, y: &Array2<V>, teacher: &Array2<V>) -> Array2<V>
    where
        V: Float + ScalarOperand,
    {
        let batch_size =
            V::from(teacher.dim().0).expect("Failed to cast usize into the value type");
        (y - teacher.to_owned()) / batch_size
    }
}

impl<V> Terminal<V, Ix2> for SoftmaxWithLoss
where
    V: Float + ScalarOperand,
{
    fn predict(&self, input: &Array2<V>) -> Array2<V> {
        input.to_owned()
    }

    fn loss(&self, input: &Array2<V>, teacher: &Array2<V>) -> V {
        self.loss(input, teacher).1
    }

    fn fit(&mut self, input: &Array2<V>, teacher: &Array2<V>) -> Array2<V> {
        let (y, _) = self.loss(input, teacher);
        self.backward(&y, teacher)
    }
}
