pub mod layers;
pub mod network;
pub mod optimizers;
pub mod output_layers;
pub mod traits;
mod utils;

#[cfg(test)]
mod mnist_demo {
    use mnist::{MnistBuilder, NormalizedMnist};
    use ndarray::{Array2, Axis};
    use rand::{seq::SliceRandom, Rng};

    use crate::{
        layers::{Add, Dot, Relu, Softmax},
        network::Network,
        optimizers::Sgd,
        output_layers::CrossEntropy,
        traits::Terminal,
    };

    const TRAINING_LEN: usize = 60000;
    const TEST_LEN: usize = 10000;
    const ITERS_NUM: usize = 10000;
    const BATCH_SIZE: usize = 100;
    const LEARNING_RATE: f32 = 0.1;
    const WGT_INIT_STD: f32 = 0.01;

    #[test]
    fn demo() {
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
                let trn_acc = net.accuracy_by_key(&x_trn, &t_trn, |data| max_position(data));
                let test_acc = net.accuracy_by_key(&x_test, &t_test, |data| max_position(data));
                trn_acc_list.push(trn_acc);
                test_acc_list.push(test_acc);
                println!("{}, {}", trn_acc, test_acc);
            }
        }
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
            Dot::new_random((input_size, hidden_size), Sgd::new(LEARNING_RATE), rng, WGT_INIT_STD),
            Add::new_zero(hidden_size, Sgd::new(LEARNING_RATE)),
            Relu::new(),
            Dot::new_random((hidden_size, output_size), Sgd::new(LEARNING_RATE), rng, WGT_INIT_STD),
            Add::new_zero(output_size, Sgd::new(LEARNING_RATE)),
            Softmax::new();
            CrossEntropy::new()
        )
    }

    fn max_position<V, I>(x: I) -> Option<usize>
    where
        V: PartialOrd,
        I: IntoIterator<Item = V>,
    {
        x.into_iter()
            .enumerate()
            .max_by(|(_, xi), (_, xj)| xi.partial_cmp(xj).expect("Comparing failed"))
            .map(|(i, _)| i)
    }
}
