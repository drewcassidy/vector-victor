use crate::{Col, Mat};
use num_traits::Num;
use std::num::{NonZero, Saturating, Wrapping};

pub trait Scalar {}

// impl<T> Scalar for T where T: Num {}

macro_rules! multi_impl {
    ($name:ident for $($t:ty),*) => ($(
        impl $name for $t {}
    )*)
}

multi_impl!(Scalar for i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize, f32, f64, bool);

multi_impl!(Scalar for NonZero<i8>, NonZero<i16>, NonZero<i32>, NonZero<i64>, NonZero<i128>, NonZero<isize>, NonZero<u8>, NonZero<u16>, NonZero<u32>, NonZero<u64>, NonZero<u128>, NonZero<usize>);

impl<T: Scalar> Scalar for Wrapping<T> {}
impl<T: Scalar> Scalar for Saturating<T> {}

pub trait Splat<T: Copy> {
    fn splat(self) -> T;
}

/// Identity splat
impl<T: Copy> Splat<T> for T {
    fn splat(self) -> T {
        self
    }
}

/// Identity splat (Reference)
impl<T: Copy> Splat<T> for &T {
    fn splat(self) -> T {
        *self
    }
}

/// Scalar to Vector Splat
impl<T: Copy, R: Scalar + Splat<T>, const N: usize> Splat<Col<T, N>> for R {
    fn splat(self) -> Col<T, N> {
        Col::<T, N> {
            data: [self.splat(); N],
        }
    }
}

/// Vector to Matrix Splat
impl<T: Copy + Splat<Col<T, W>>, const H: usize, const W: usize> Splat<Mat<T, H, W>> for Col<T, H> {
    fn splat(self) -> Mat<T, H, W> {
        self.map(|row| row.splat())
    }
}

/// Vector to Matrix Splat (Reference)
impl<T: Copy + Splat<Col<T, W>>, const H: usize, const W: usize> Splat<Mat<T, H, W>>
    for &Col<T, H>
{
    fn splat(self) -> Mat<T, H, W> {
        self.map(|row| row.splat())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Mat;

    #[test]
    fn test_splat_scalar_to_col() {
        let a: Col<i32, 4> = 5.splat();
        assert_eq!(a, Col::from([5, 5, 5, 5]))
    }

    #[test]
    fn test_splat_col_to_matrix() {
        let a = Col::from([1, 2, 3]);
        let b: Mat<_, 3, 4> = a.splat();
        for n in 0..3 {
            assert_eq!(b.data[n], Col::from([a.data[n]; 4]))
        }
    }

    #[test]
    fn test_splat_scalar_to_matrix() {
        let a: Mat<_, 3, 4> = 5.splat();
        for n in 0..3 {
            assert_eq!(a.data[n], Col::from([5; 4]));
        }
    }
}
