use ndarray::{Array1, Array2, Axis, Ix1, LinalgScalar, Ix2};
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

impl<V, O> Layer<V, Ix2> for Add<V, O>
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

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2, Ix1};

    use crate::traits::{Layer, Optimizer};

    use super::Add;

    fn bias() -> Array1<usize> {
        Array1::from_vec(vec![1, 2, 3])
    }
    fn input() -> Array2<usize> {
        Array2::from_shape_vec((2, 3), vec![4, 5, 6, 7, 8, 9]).unwrap()
    }
    fn grad() -> Array2<usize> {
        Array2::from_shape_vec((2, 3), vec![10, 11, 12, 13, 14, 15]).unwrap()
    }
    fn bias_after_update() -> Array1<usize> {
        Array1::from_vec(vec![16, 17, 18])
    }

    struct DummyOpt;
    impl Optimizer<usize, Ix1> for DummyOpt {
        fn update(
            &mut self,
            param: &mut ndarray::Array<usize, Ix1>,
            _: ndarray::Array<usize, Ix1>,
        ) {
            *param = bias_after_update()
        }
    }

    #[test]
    fn should_add_bias_to_input() {
        let expected =
            Array2::from_shape_vec((2, 3), vec![4 + 1, 5 + 2, 6 + 3, 7 + 1, 8 + 2, 9 + 3]).unwrap();

        let layer = Add::new(bias(), DummyOpt);
        let result = layer.forward(&input());

        assert_eq!(result, expected)
    }

    #[test]
    fn should_return_the_given_gradient() {
        let layer = Add::new(bias(), DummyOpt);
        let result = layer.backward(&grad(), &input());

        assert_eq!(result, grad())
    }

    #[test]
    fn should_call_optimizer_update() {
        let mut layer = Add::new(bias(), DummyOpt);
        layer.learn(&grad(), &input());

        assert_eq!(layer.bias, bias_after_update())
    }

    #[test]
    fn should_initialize_bias_with_zero() {
        let layer = Add::<usize, _>::new_zero(3, DummyOpt);
        assert_eq!(layer.bias, Array1::zeros(3))
    }
}
