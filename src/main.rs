use std::ops::{Mul, SubAssign};

use mnist::{MnistBuilder, NormalizedMnist};
use ndarray::{Array, Array1, Array2, Axis, Dim, Dimension, LinalgScalar, ScalarOperand};
use num_traits::{Float, Zero};
use rand::{distributions::Uniform, seq::SliceRandom, Rng};

const TRAINING_LEN: usize = 60000;
const TEST_LEN: usize = 10000;
const ITERS_NUM: usize = 10000;
const BATCH_SIZE: usize = 100;
const LEARNING_RATE: f32 = 0.1;
const WGT_INIT_STD: f32 = 0.01;

fn main() {
    let NormalizedMnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .download_and_extract()
        .training_set_length(TRAINING_LEN as u32)
        .test_set_length(TEST_LEN as u32)
        .label_format_one_hot()
        .finalize()
        .normalize();

    let x_trn = Array2::from_shape_vec((TRAINING_LEN, 28 * 28), trn_img)
        .expect("Error converting images to Array2 struct");
    let t_trn = Array2::from_shape_vec((TRAINING_LEN, 10), trn_lbl)
        .expect("Error converting training labels to Array2 struct")
        .mapv(|v| v as f32);
    let x_test = Array2::from_shape_vec((TEST_LEN, 28 * 28), tst_img)
        .expect("Error converting images to Array2 struct");
    let t_test = Array2::from_shape_vec((TEST_LEN, 10), tst_lbl)
        .expect("Error converting test labels to Array2 struct")
        .mapv(|v| v as f32);

    let mut trn_loss_list = vec![];
    let mut trn_acc_list = vec![];
    let mut test_acc_list = vec![];

    let iter_per_epoch = (TRAINING_LEN / BATCH_SIZE).max(1);

    let mut rng = rand::thread_rng();
    let mut net = make_two_layer_net(28 * 28, 50, 10, &mut rng);
    let all_indices = (0..x_trn.dim().0).collect::<Vec<_>>();

    for i in 0..ITERS_NUM {
        let indices = all_indices
            .choose_multiple(&mut rng, BATCH_SIZE)
            .copied()
            .collect::<Vec<_>>();
        let x_batch = x_trn.select(Axis(0), &indices);
        let t_batch = t_trn.select(Axis(0), &indices);

        net.fit(&x_batch, &t_batch);
        trn_loss_list.push(net.loss(&x_batch, &t_batch));

        if i % iter_per_epoch == 0 {
            let trn_acc = net.accuracy(&x_trn, &t_trn);
            let test_acc = net.accuracy(&x_test, &t_test);
            trn_acc_list.push(trn_acc);
            test_acc_list.push(test_acc);
            println!("{}, {}", trn_acc, test_acc);
        }
    }
}

fn random_matrix<R: Rng>(shape: (usize, usize), rng: &mut R) -> Array2<f32> {
    WGT_INIT_STD
        * Array2::from_shape_vec(
            shape,
            rng.sample_iter(Uniform::new(0., 1.))
                .take(shape.0 * shape.1)
                .collect(),
        )
        .unwrap()
}

macro_rules! network {
    ( $head:expr; $last_layer:expr ) => {
        Network::new(Box::new($head), Box::new($last_layer))
    };
    ( $head:expr, $( $tail:expr ),+ ; $last_layer:expr ) => {
        Network::new(
            Box::new($head),
            Box::new(network!($($tail),* ; $last_layer))
        )
    };
}

fn make_two_layer_net<R: Rng>(
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    rng: &mut R,
) -> Network<f32> {
    network!(
        Dot::new(random_matrix((input_size, hidden_size), rng), Sgd::new(LEARNING_RATE)),
        Add::new(Array1::zeros(hidden_size), Sgd::new(LEARNING_RATE)),
        Relu::new(),
        Dot::new(random_matrix((hidden_size, output_size), rng), Sgd::new(LEARNING_RATE)),
        Add::new(Array1::zeros(output_size), Sgd::new(LEARNING_RATE)),
        Softmax::new();
        CrossEntropy::new()
    )
}

trait Layer<Input, Output> {
    fn forward(&self, input: &Input) -> Output;
    fn backward(&self, grad_out: &Output, input: &Input) -> Input;
    fn learn(&mut self, grad_out: &Output, input: &Input);
}

trait Fittable<Input, Output, Loss> {
    fn predict(&self, input: &Input) -> Output;
    fn loss(&self, input: &Input, teacher: &Output) -> Loss;
    fn fit(&mut self, input: &Input, teacher: &Output) -> Input;
}

type FMat = Array2<f32>;

struct Network<V> {
    head: Box<dyn Layer<Array2<V>, Array2<V>>>,
    tail: Box<dyn Fittable<Array2<V>, Array2<V>, V>>,
}

impl<V> Network<V> {
    fn new(
        head: Box<dyn Layer<Array2<V>, Array2<V>>>,
        tail: Box<dyn Fittable<Array2<V>, Array2<V>, V>>,
    ) -> Self {
        Self { head, tail }
    }
}

impl<V> Network<V>
where
    V: PartialOrd,
{
    fn accuracy(&mut self, input: &Array2<V>, teacher: &Array2<V>) -> f32 {
        let y = self.predict(input);

        let a = teacher
            .axis_iter(Axis(0))
            .map(|data| {
                data.iter()
                    .enumerate()
                    .max_by(|(_, v), (_, w)| v.partial_cmp(w).unwrap())
                    .unwrap()
                    .0
            })
            .collect::<Vec<_>>();

        let count = a
            .into_iter()
            .zip(y.axis_iter(Axis(0)))
            .filter(|(t_max, yi)| {
                let y_max = yi
                    .iter()
                    .enumerate()
                    .max_by(|(_, v), (_, w)| v.partial_cmp(w).unwrap())
                    .map(|(i, _)| i);
                *t_max == y_max.unwrap()
            })
            .count();
        count as f32 / input.dim().0 as f32
    }
}

impl<V> Fittable<Array2<V>, Array2<V>, V> for Network<V> {
    fn predict(&self, input: &Array2<V>) -> Array2<V> {
        self.tail.predict(&self.head.forward(input))
    }

    fn loss(&self, input: &Array2<V>, teacher: &Array2<V>) -> V {
        self.tail.loss(&self.head.forward(input), teacher)
    }

    fn fit(&mut self, input: &Array2<V>, teacher: &Array2<V>) -> Array2<V> {
        let y = self.head.forward(input);
        let dy = self.tail.fit(&y, teacher);
        let dx = self.head.backward(&dy, input);
        self.head.learn(&dy, input);
        dx
    }
}

trait Optimizer<A, D> {
    fn update(&mut self, param: &mut Array<A, D>, grad: Array<A, D>);
}

struct Sgd<V> {
    learning_rate: V,
}

impl<V> Sgd<V> {
    fn new(learning_rate: V) -> Self {
        Self { learning_rate }
    }
}

impl<V, D> Optimizer<V, D> for Sgd<V>
where
    D: Dimension,
    V: Mul<Array<V, D>, Output = Array<V, D>> + SubAssign + Clone,
{
    fn update(&mut self, param: &mut Array<V, D>, grad: Array<V, D>) {
        *param -= &(self.learning_rate.clone() * grad);
    }
}

struct Dot<V, O: Optimizer<V, Dim<[usize; 2]>>> {
    wgt: Array2<V>,
    optimizer: O,
}

impl<V, O: Optimizer<V, Dim<[usize; 2]>>> Dot<V, O> {
    fn new(wgt: Array2<V>, optimizer: O) -> Self {
        Self { wgt, optimizer }
    }
}

impl<V, O: Optimizer<V, Dim<[usize; 2]>>> Layer<Array2<V>, Array2<V>> for Dot<V, O>
where
    V: LinalgScalar,
{
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

struct Add<V, O: Optimizer<V, Dim<[usize; 1]>>> {
    bias: Array1<V>,
    optimizer: O,
}

impl<V, O: Optimizer<V, Dim<[usize; 1]>>> Add<V, O> {
    fn new(bias: Array1<V>, optimizer: O) -> Self {
        Self { bias, optimizer }
    }
}

impl<V, O> Layer<Array2<V>, Array2<V>> for Add<V, O>
where
    V: LinalgScalar,
    O: Optimizer<V, Dim<[usize; 1]>>,
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

struct Relu {}

impl Relu {
    fn new() -> Self {
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

struct Softmax {}

impl Softmax {
    fn new() -> Self {
        Self {}
    }
}

impl Layer<FMat, FMat> for Softmax {
    fn forward(&self, input: &FMat) -> FMat {
        softmax(input)
    }

    fn backward(&self, grad_out: &FMat, input: &FMat) -> FMat {
        let output = self.forward(input);

        let dot = output
            .axis_iter(Axis(0))
            .zip(grad_out.axis_iter(Axis(0)))
            .map(|(out, grad)| out.dot(&grad))
            .collect::<Array1<_>>()
            .into_shape((output.dim().0, 1))
            .unwrap();

        output * (grad_out - &dot)
    }

    fn learn(&mut self, _: &FMat, _: &FMat) {}
}

struct CrossEntropy {}

impl CrossEntropy {
    fn new() -> Self {
        Self {}
    }
}

impl Fittable<FMat, FMat, f32> for CrossEntropy {
    fn predict(&self, input: &FMat) -> FMat {
        input.to_owned()
    }

    fn loss(&self, input: &FMat, teacher: &FMat) -> f32 {
        cross_entropy_error(input, teacher)
    }

    fn fit(&mut self, input: &FMat, teacher: &FMat) -> FMat {
        let batch_size = teacher.dim().0;
        -teacher / input / (batch_size as f32)
    }
}

struct SoftmaxWithLoss {}

impl SoftmaxWithLoss {
    fn new() -> Self {
        Self {}
    }

    fn loss(&self, input: &Array2<f32>, teacher: &Array2<f32>) -> (Array2<f32>, f32) {
        let y = softmax(input);
        let error = cross_entropy_error(&y, teacher);
        (y, error)
    }

    fn backward(&self, y: &Array2<f32>, teacher: &Array2<f32>) -> Array2<f32> {
        let batch_size = teacher.dim().0;
        (y - teacher) / (batch_size as f32)
    }
}

impl Fittable<FMat, FMat, f32> for SoftmaxWithLoss {
    fn predict(&self, input: &FMat) -> FMat {
        input.to_owned()
    }

    fn loss(&self, input: &FMat, teacher: &FMat) -> f32 {
        self.loss(input, teacher).1
    }

    fn fit(&mut self, input: &Array2<f32>, teacher: &Array2<f32>) -> Array2<f32> {
        let (y, _) = self.loss(input, teacher);
        self.backward(&y, teacher)
    }
}

fn softmax<V>(x: &Array2<V>) -> Array2<V>
where
    V: Float + ScalarOperand,
{
    x.axis_iter(Axis(0))
        .flat_map(|row| {
            let max = *row
                .iter()
                .max_by(|a, b| a.partial_cmp(b).expect("NaN found"))
                .expect("Data is empty");
            let exp_row = (&row - max).mapv(|v| v.exp());
            let exp_sum = exp_row.iter().fold(V::zero(), |a, b| a + *b);
            exp_row / exp_sum
        })
        .collect::<Array1<_>>()
        .into_shape(x.dim())
        .expect("Failed to collect values into an array")
}

fn cross_entropy_error<V>(y: &Array2<V>, t: &Array2<V>) -> V
where
    V: Float,
{
    y.iter()
        .zip(t)
        .map(|(yi, ti)| -(*ti) * yi.ln())
        .fold(V::zero(), |a, b| a + b)
}
