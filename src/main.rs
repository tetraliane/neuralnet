use mnist::{MnistBuilder, NormalizedMnist};
use ndarray::{Array1, Array2, Axis};
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
    let mut net = TwoLayerNet::new(28 * 28, 50, 10, &mut rng);
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

struct TwoLayerNet {
    layers: Connection,
}

impl TwoLayerNet {
    fn new<R: Rng>(input_size: usize, hidden_size: usize, output_size: usize, rng: &mut R) -> Self {
        let w1 = random_matrix((input_size, hidden_size), rng);
        let b1 = Array1::zeros(hidden_size);
        let w2 = random_matrix((hidden_size, output_size), rng);
        let b2 = Array1::zeros(output_size);

        let layers = Connection::new(
            Box::new(Dot::new(w1)),
            Box::new(Connection::new(
                Box::new(Add::new(b1)),
                Box::new(Connection::new(
                    Box::new(Relu::new()),
                    Box::new(Connection::new(
                        Box::new(Dot::new(w2)),
                        Box::new(Connection::new(
                            Box::new(Add::new(b2)),
                            Box::new(SoftmaxWithLoss::new()),
                        )),
                    )),
                )),
            )),
        );

        Self { layers }
    }

    fn fit(&mut self, input: &Array2<f32>, teacher: &Array2<f32>) {
        self.layers.fit(input, teacher);
    }

    fn loss(&mut self, input: &Array2<f32>, teacher: &Array2<f32>) -> f32 {
        self.layers.loss(input, teacher)
    }

    fn predict(&mut self, input: &Array2<f32>) -> Array2<f32> {
        self.layers.predict(input)
    }

    fn accuracy(&mut self, input: &Array2<f32>, teacher: &Array2<f32>) -> f32 {
        self.layers.accuracy(input, teacher)
    }
}

trait Layer<Input, Output> {
    fn forward(&self, x: &Input) -> Output;
    fn backward(&self, dout: &Output, input: &Input) -> Input;
    fn learn(&mut self, dout: &Output, x: &Input);
}

trait Fittable<Input, Output, Loss> {
    fn predict(&self, input: &Input) -> Output;
    fn loss(&self, input: &Input, teacher: &Output) -> Loss;
    fn fit(&mut self, input: &Input, teacher: &Output) -> Input;
}

type FMat = Array2<f32>;

struct Connection {
    head: Box<dyn Layer<FMat, FMat>>,
    tail: Box<dyn Fittable<FMat, FMat, f32>>,
}

impl Connection {
    fn new(head: Box<dyn Layer<FMat, FMat>>, tail: Box<dyn Fittable<FMat, FMat, f32>>) -> Self {
        Self { head, tail }
    }

    fn accuracy(&mut self, input: &Array2<f32>, teacher: &Array2<f32>) -> f32 {
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

impl Fittable<FMat, FMat, f32> for Connection {
    fn predict(&self, input: &FMat) -> FMat {
        self.tail.predict(&self.head.forward(input))
    }

    fn loss(&self, input: &FMat, teacher: &FMat) -> f32 {
        self.tail.loss(&self.head.forward(input), teacher)
    }

    fn fit(&mut self, input: &FMat, teacher: &FMat) -> FMat {
        let y = self.head.forward(&input);
        let dy = self.tail.fit(&y, teacher);
        let dx = self.head.backward(&dy, &input);
        self.head.learn(&dy, &input);
        dx
    }
}

#[derive(Clone)]
struct Dot {
    wgt: FMat,
}

impl Dot {
    fn new(wgt: FMat) -> Self {
        Self { wgt }
    }
}

impl Layer<FMat, FMat> for Dot {
    fn forward(&self, input: &FMat) -> FMat {
        input.dot(&self.wgt)
    }

    fn backward(&self, dout: &FMat, _: &FMat) -> FMat {
        dout.dot(&self.wgt.t())
    }

    fn learn(&mut self, dout: &FMat, input: &FMat) {
        self.wgt -= &(LEARNING_RATE * input.t().dot(dout));
    }
}

#[derive(Clone)]
struct Add {
    bias: Array1<f32>,
}

impl Add {
    fn new(bias: Array1<f32>) -> Self {
        Self { bias }
    }
}

impl Layer<FMat, FMat> for Add {
    fn forward(&self, input: &FMat) -> FMat {
        input + &self.bias
    }

    fn backward(&self, dout: &FMat, _: &FMat) -> FMat {
        dout.to_owned()
    }

    fn learn(&mut self, dout: &FMat, _: &FMat) {
        self.bias -= &(LEARNING_RATE * dout.sum_axis(Axis(0)));
    }
}

// #[derive(Clone)]
// struct Affine {
//     dot_layer: Dot,
//     add_layer: Add,
//     dot_input: Option<FMat>,
//     dot_dout: Option<FMat>,
//     add_input: Option<FMat>,
//     add_dout: Option<FMat>,
// }

// impl Affine {
//     fn new(wgt: Array2<f32>, bias: Array1<f32>) -> Self {
//         Self {
//             dot_layer: Dot::new(wgt),
//             add_layer: Add::new(bias),
//             dot_input: None,
//             dot_dout: None,
//             add_input: None,
//             add_dout: None,
//         }
//     }

//     fn fit(&mut self, input: FMat, dout: FMat) -> FMat {
//         self.forward(input.clone());
//         let backward_result = self.backward(dout.clone(), &input);
//         self.learn(&dout, &input);
//         backward_result
//     }
// }

// impl Layer<Array2<f32>, Array2<f32>> for Affine {
//     fn forward(&mut self, x: Array2<f32>) -> Array2<f32> {
//         self.dot_input = Some(x.clone());
//         let dot = self.dot_layer.forward(x);

//         self.add_input = Some(dot.clone());
//         self.add_layer.forward(dot)
//     }
//     fn backward(&mut self, dout: Array2<f32>, _: &FMat) -> Array2<f32> {
//         self.add_dout = Some(dout.clone());
//         let add_input = self.add_input.as_ref().unwrap();
//         let dout = self.add_layer.backward(dout, add_input);

//         self.dot_dout = Some(dout.clone());
//         let dot_input = self.dot_input.as_ref().unwrap();
//         self.dot_layer.backward(dout, dot_input)
//     }

//     fn learn(&mut self, _: &FMat, _: &FMat) {
//         self.dot_layer.learn(
//             self.dot_dout.as_ref().unwrap(),
//             self.dot_input.as_ref().unwrap(),
//         );
//         self.add_layer.learn(
//             self.add_dout.as_ref().unwrap(),
//             self.add_input.as_ref().unwrap(),
//         );
//     }
// }

#[derive(Clone)]
struct Relu {}

impl Relu {
    fn new() -> Self {
        Self {}
    }
}

impl Layer<Array2<f32>, Array2<f32>> for Relu {
    fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        x.map(|xi| xi.max(0.))
    }

    fn backward(&self, dout: &Array2<f32>, input: &FMat) -> Array2<f32> {
        let mut dx = dout.to_owned();
        dx.zip_mut_with(input, |dout_i, xi| {
            *dout_i = if *xi <= 0. { 0. } else { *dout_i }
        });
        dx
    }

    fn learn(&mut self, _: &Array2<f32>, _: &Array2<f32>) {}
}

#[derive(Clone)]
struct SoftmaxWithLoss {}

impl SoftmaxWithLoss {
    fn new() -> Self {
        Self {}
    }

    fn loss(&self, x: &Array2<f32>, teacher: &Array2<f32>) -> (Array2<f32>, f32) {
        let y = softmax(x);
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
        self.loss(input, &teacher).1
    }

    fn fit(&mut self, input: &Array2<f32>, teacher: &Array2<f32>) -> Array2<f32> {
        let (y, _) = self.loss(input, teacher);
        self.backward(&y, teacher)
    }
}

fn softmax(x: &Array2<f32>) -> Array2<f32> {
    let mut values = vec![];
    for row in x.axis_iter(Axis(0)) {
        let max = *row.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let a = &row - Array1::<f32>::ones(row.dim()) * max;
        let exp_row = a.mapv(|v| v.exp());
        let exp_sum = exp_row.iter().sum::<f32>();
        values.push((exp_row / exp_sum).to_vec())
    }
    Array2::from_shape_vec(x.dim(), values.into_iter().flatten().collect()).unwrap()
}

fn cross_entropy_error(y: &Array2<f32>, t: &Array2<f32>) -> f32 {
    y.iter().zip(t).map(|(yi, ti)| -1. * (*ti) * yi.ln()).sum()
}
