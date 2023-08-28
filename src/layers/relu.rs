use ndarray::Array2;
use num_traits::Zero;

use crate::traits::Layer;

pub struct Relu {}

impl Relu {
    pub fn new() -> Self {
        Self {}
    }
}

impl<V> Layer<Array2<V>, Array2<V>> for Relu
where
    V: PartialOrd + Clone + Zero,
{
    fn forward(&self, input: &Array2<V>) -> Array2<V> {
        input.map(|xi| {
            if xi > &V::zero() {
                xi.clone()
            } else {
                V::zero()
            }
        })
    }

    fn backward(&self, grad_out: &Array2<V>, input: &Array2<V>) -> Array2<V> {
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

    fn learn(&mut self, _: &Array2<V>, _: &Array2<V>) {}
}
