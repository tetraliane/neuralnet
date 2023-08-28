use ndarray::Array;

pub trait Layer<Input, Output> {
    fn forward(&self, input: &Input) -> Output;
    fn backward(&self, grad_out: &Output, input: &Input) -> Input;
    fn learn(&mut self, grad_out: &Output, input: &Input);
}

pub trait Terminal<Input, Output, Loss> {
    fn predict(&self, input: &Input) -> Output;
    fn loss(&self, input: &Input, teacher: &Output) -> Loss;
    fn fit(&mut self, input: &Input, teacher: &Output) -> Input;
}

pub trait Optimizer<A, D> {
    fn update(&mut self, param: &mut Array<A, D>, grad: Array<A, D>);
}
