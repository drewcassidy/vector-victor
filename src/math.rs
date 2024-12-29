use crate::Col;
use num_traits::{One, Zero};
use std::iter::{Product, Sum};
use std::ops::{Add, Mul};

pub trait Dot<Rhs = Self> {
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

pub trait MMul<Rhs = Self> {
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
    use super::*;
    use crate::Mat;

    #[test]
    fn test_dot() {
        let a = Col::from([1, 2, 3]);
        let b = Col::from([3, 4, 5]);
        let c = a.dot(b);
        assert_eq!(c, 26)
    }

    #[test]
    fn test_ident() {
        let a = Mat::from([[1, 0], [0, 1]]);
        let b = Mat::from([[3, 4], [5, 6]]);
        assert_eq!(a.mmul(b), b)
    }

    #[test]
    fn test_mul() {
        let a = Mat::from([[3, 4], [5, 6]]);
        let b = Col::from([1, 2]);
        assert_eq!(a * b, Mat::from([[3, 4], [10, 12]]))
    }
}
