use ndarray::{Array, Dimension};
use num_traits::Zero;

use crate::traits::Layer;

pub struct Relu {}

impl Relu {
    pub fn new() -> Self {
        Self {}
    }
}

impl<V, D> Layer<V, D> for Relu
where
    V: PartialOrd + Clone + Zero,
    D: Dimension,
{
    fn params(&self) -> Option<Array<V, ndarray::IxDyn>> {
        None
    }

    fn forward(&self, input: &Array<V, D>) -> Array<V, D> {
        input.map(|xi| {
            if xi > &V::zero() {
                xi.clone()
            } else {
                V::zero()
            }
        })
    }

    fn backward(&self, grad_out: &Array<V, D>, input: &Array<V, D>) -> Array<V, D> {
        let mut dx = grad_out.to_owned();
        dx.zip_mut_with(input, |dout_i, xi| {
            *dout_i = if *xi <= V::zero() {
                V::zero()
            } else {
                dout_i.clone()
            }
        });
        dx
    }

    fn learn(&mut self, _: &Array<V, D>, _: &Array<V, D>) {}
}
