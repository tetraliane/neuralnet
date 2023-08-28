use ndarray::{Array1, Array2, Axis, ScalarOperand};
use num_traits::Float;

pub(crate) fn softmax<V>(x: &Array2<V>) -> Array2<V>
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

pub(crate) fn cross_entropy_error<V>(y: &Array2<V>, t: &Array2<V>) -> V
where
    V: Float,
{
    y.iter()
        .zip(t)
        .map(|(yi, ti)| -(*ti) * yi.ln())
        .fold(V::zero(), |a, b| a + b)
}
