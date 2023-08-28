use ndarray::Array;

pub trait Layer<T, D> {
    fn forward(&self, input: &Array<T, D>) -> Array<T, D>;
    fn backward(&self, grad_out: &Array<T, D>, input: &Array<T, D>) -> Array<T, D>;
    fn learn(&mut self, grad_out: &Array<T, D>, input: &Array<T, D>);
}

pub trait Terminal<T, D> {
    fn predict(&self, input: &Array<T, D>) -> Array<T, D>;
    fn loss(&self, input: &Array<T, D>, teacher: &Array<T, D>) -> T;
    fn fit(&mut self, input: &Array<T, D>, teacher: &Array<T, D>) -> Array<T, D>;

    fn layer_at(&self, _: usize) -> Option<&dyn Layer<T, D>> {
        None
    }
    fn layer_mut_at(&mut self, _: usize) -> Option<&mut dyn Layer<T, D>> {
        None
    }
}

pub trait Optimizer<A, D> {
    fn update(&mut self, param: &mut Array<A, D>, grad: Array<A, D>);
}
