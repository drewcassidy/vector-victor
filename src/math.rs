use crate::Col;
use num_traits::Zero;
use std::iter::{zip, Sum};
use std::ops::{Add, Mul};

impl<T: Copy, const N: usize> Sum for Col<T, N>
where
    T: Zero + Add<T, Output = T>,
{
    fn sum<I: Iterator<Item = Self>>(mut iter: I) -> Self {
        let acc = iter.next().unwrap_or(Col::from([T::zero(); N]));
        iter.fold(acc, std::ops::Add::add)
    }
}

trait Dot<Rhs = Self> {
    type Output;

    fn dot(self, rhs: Rhs) -> Self::Output;
}

impl<L, R, const N: usize> Dot<Col<R, N>> for Col<L, N>
where
    L: Mul<R, Output: Sum + Copy> + Copy,
    R: Copy,
{
    type Output = <L as Mul<R>>::Output;

    fn dot(self, rhs: Col<R, N>) -> Self::Output {
        zip(self.rows(), rhs.rows()).map(|(l, r)| l * r).sum()
    }
}

impl<L: Copy, R: Copy, const N: usize> Dot<&Col<R, N>> for Col<L, N>
where
    Col<L, N>: Dot<Col<R, N>>,
{
    type Output = <Col<L, N> as Dot<Col<R, N>>>::Output;

    fn dot(self, rhs: &Col<R, N>) -> Self::Output {
        self.dot(*rhs)
    }
}

trait MMul<Rhs = Self> {
    type Output;

    fn mmul(self, rhs: Rhs) -> Self::Output;
}

impl<L: Copy, R: Copy, const N: usize> MMul<Col<R, N>> for Col<L, N>
where
    Col<R, N>: Dot<L, Output: Copy>,
{
    type Output = Col<<Col<R, N> as Dot<L>>::Output, N>;

    fn mmul(self, rhs: Col<R, N>) -> Self::Output {
        Self::Output::try_from_rows(self.rows().map(|l| rhs.dot(l))).expect("Length mismatch")
    }
}

impl<L: Copy, R: Copy, const N: usize> MMul<&Col<R, N>> for Col<L, N>
where
    Col<L, N>: MMul<Col<R, N>>,
{
    type Output = <Col<L, N> as MMul<Col<R, N>>>::Output;

    fn mmul(self, rhs: &Col<R, N>) -> Self::Output {
        self.mmul(*rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::Dot;
    use super::*;

    #[test]
    fn test_dot() {
        let a = Col::from([1, 2, 3]);
        let b = Col::from([3, 4, 5]);
        let c = a.dot(b);
        assert_eq!(c, 26)
    }

    #[test]
    fn test_ident() {
        let a = Col::from([Col::from([1, 0]), Col::from([0, 1])]);
        let b = Col::from([Col::from([3, 4]), Col::from([5, 6])]);
        assert_eq!(a.mmul(b), b)
    }
}
