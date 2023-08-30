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

#[cfg(test)]
mod tests {
    use ndarray::{Array2, Ix2};

    use crate::traits::{Layer, Optimizer};

    use super::Dot;

    fn wgt() -> Array2<usize> {
        Array2::from_shape_vec((2, 2), vec![1, 2, 3, 4]).unwrap()
    }
    fn input() -> Array2<usize> {
        Array2::from_shape_vec((3, 2), vec![7, 8, 9, 10, 11, 12]).unwrap()
    }
    fn grad() -> Array2<usize> {
        Array2::from_shape_vec((3, 2), vec![11, 12, 13, 14, 15, 16]).unwrap()
    }
    fn wgt_after_update() -> Array2<usize> {
        Array2::from_shape_vec((2, 2), vec![17, 18, 19, 20]).unwrap()
    }

    struct DummyOpt;
    impl Optimizer<usize, Ix2> for DummyOpt {
        fn update(
            &mut self,
            param: &mut ndarray::Array<usize, Ix2>,
            _: ndarray::Array<usize, Ix2>,
        ) {
            *param = wgt_after_update()
        }
    }
    impl Optimizer<f32, Ix2> for DummyOpt {
        fn update(&mut self, _: &mut ndarray::Array<f32, Ix2>, _: ndarray::Array<f32, Ix2>) {}
    }

    #[test]
    fn should_have_given_array_as_params() {
        let layer = Dot::new(wgt(), DummyOpt);
        assert_eq!(layer.params().unwrap(), wgt().into_dyn())
    }

    #[test]
    fn should_calculate_dot_product() {
        let layer = Dot::new(wgt(), DummyOpt);
        let result = layer.forward(&input());

        let expected = input().dot(&wgt());
        assert_eq!(result, expected)
    }

    #[test]
    fn should_return_gradient_of_input() {
        let layer = Dot::new(wgt(), DummyOpt);
        let result = layer.backward(&grad(), &input());

        let expected = grad().dot(&wgt().t());
        assert_eq!(result, expected)
    }

    #[test]
    fn should_call_optimizer_update() {
        let mut layer = Dot::new(wgt(), DummyOpt);
        layer.learn(&grad(), &input());

        assert_eq!(layer.params().unwrap(), wgt_after_update().into_dyn());

        let expected = input().dot(&wgt_after_update());
        assert_eq!(layer.forward(&input()), expected)
    }

    #[test]
    fn test_random_initializer() {
        let mut rng = rand::thread_rng();
        let layer = Dot::new_random((2, 2), DummyOpt, &mut rng, 1.);
        assert_eq!(layer.params().unwrap().dim(), ndarray::IxDyn(&[2, 2]));
    }
}
