use crate::Col;
use num_traits::{One, Zero};
use std::iter::{zip, Sum};
use std::ops::{Add, Mul};

impl<T: Copy, const N: usize> Sum for Col<T, N>
where
    Col<T, N>: Zero + Add<Col<T, N>, Output = Col<T, N>>,
{
    fn sum<I: Iterator<Item = Self>>(mut iter: I) -> Self {
        let acc = iter.next().unwrap_or(Zero::zero());
        iter.fold(acc, Add::add)
    }
}

trait Dot<Rhs = Self> {
    type Output;

    fn dot(self, rhs: Rhs) -> Self::Output;
}

impl<L: Copy, R: Copy, const N: usize> Dot<R> for Col<L, N>
where
    Col<L, N>: Mul<R, Output: IntoIterator<Item: Sum + Copy>>,
{
    type Output = <<Col<L, N> as Mul<R>>::Output as IntoIterator>::Item;

    fn dot(self, rhs: R) -> Self::Output {
        let prod = self * rhs;
        prod.into_iter().sum()
    }
}

trait MMul<Rhs = Self> {
    type Output;

    fn mmul(self, rhs: Rhs) -> Self::Output;
}

impl<L: Copy, R: Copy, const N: usize> MMul<R> for Col<L, N>
where
    R: Dot<L, Output: Copy>,
{
    type Output = Col<<R as Dot<L>>::Output, N>;

    fn mmul(self, rhs: R) -> Self::Output {
        self.map(|l| rhs.dot(l))
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
        let c = a.mmul(b);
        assert_eq!(a.mmul(b), b)
    }

    #[test]
    fn test_mul() {
        let a = Col::from([1, 0]);
        let b = Col::from([Col::from([3, 4]), Col::from([5, 6])]);
        assert_eq!(b * a, Col::from([Col::from([3, 4]), Col::from([0, 0])]))
    }
}
