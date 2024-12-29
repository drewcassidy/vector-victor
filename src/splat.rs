use crate::Col;
use num_traits::Num;

pub trait Scalar {}

impl<T> Scalar for T where T: Num {}

pub trait Splat<T: Copy> {
    fn splat(self) -> T;
}

/// Identity splat
impl<T: Copy + Scalar> Splat<T> for T {
    fn splat(self) -> T {
        self
    }
}

/// Identity splat (Reference)
impl<T: Copy + Scalar> Splat<T> for &T {
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

/// Vector to Vector splat
impl<T: Copy, R: Copy + Splat<T>, const N: usize> Splat<Col<T, N>> for Col<R, N> {
    fn splat(self) -> Col<T, N> {
        self.map(Splat::splat)
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
