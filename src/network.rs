use ndarray::{Array2, ArrayView1, Axis};

use crate::traits::{Layer, Terminal};

pub struct Network<V> {
    head: Box<dyn Layer<Array2<V>, Array2<V>>>,
    tail: Box<dyn Terminal<Array2<V>, Array2<V>, V>>,
}

impl<V> Network<V> {
    pub fn new(
        head: Box<dyn Layer<Array2<V>, Array2<V>>>,
        tail: Box<dyn Terminal<Array2<V>, Array2<V>, V>>,
    ) -> Self {
        Self { head, tail }
    }
}

impl<V> Network<V>
where
    V: PartialOrd,
{
    pub fn accuracy(&mut self, input: &Array2<V>, teacher: &Array2<V>) -> f64
    where
        V: PartialEq,
    {
        self.accuracy_by(input, teacher, |y, t| y == t)
    }

    pub fn accuracy_by<F>(&mut self, input: &Array2<V>, teacher: &Array2<V>, predicate: F) -> f64
    where
        F: Fn(&ArrayView1<V>, &ArrayView1<V>) -> bool,
    {
        let y = self.predict(input);

        let count = y
            .axis_iter(Axis(0))
            .zip(teacher.axis_iter(Axis(0)))
            .filter(|(yi, ti)| predicate(yi, ti))
            .count();

        count as f64 / input.dim().0 as f64
    }

    pub fn accuracy_by_key<V2, F>(&mut self, input: &Array2<V>, teacher: &Array2<V>, f: F) -> f64
    where
        V2: PartialEq,
        F: Fn(&ArrayView1<V>) -> V2,
    {
        self.accuracy_by(input, teacher, |y, t| f(y) == f(t))
    }
}

impl<V> Terminal<Array2<V>, Array2<V>, V> for Network<V> {
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
