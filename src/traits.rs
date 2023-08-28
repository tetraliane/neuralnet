use ndarray::Array;

pub trait Layer<I> {
    type Output;
    fn forward(&self, input: &I) -> Self::Output;
    fn backward(&self, grad_out: &Self::Output, input: &I) -> I;
    fn learn(&mut self, grad_out: &Self::Output, input: &I);
}

pub trait Terminal<I> {
    type Output;
    type Loss;
    fn predict(&self, input: &I) -> Self::Output;
    fn loss(&self, input: &I, teacher: &Self::Output) -> Self::Loss;
    fn fit(&mut self, input: &I, teacher: &Self::Output) -> I;

    fn layer_at(&self, _: usize) -> Option<&dyn Layer<I, Output = I>> {
        None
    }
    fn layer_mut_at(&mut self, _: usize) -> Option<&mut dyn Layer<I, Output = I>> {
        None
    }
}

pub trait Optimizer<A, D> {
    fn update(&mut self, param: &mut Array<A, D>, grad: Array<A, D>);
}
