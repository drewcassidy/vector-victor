// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

pub mod legacy;
mod math;
mod ops;

extern crate core;

use itertools::Itertools;
pub use legacy::{Matrix, Vector};
use num_traits::{Bounded, One, Zero};
use std::cmp::min;
use std::fmt::Debug;
use std::iter::{zip, Flatten};
use std::ops::{Add, Index, IndexMut, Mul};
use std::vec::IntoIter;

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct Col<T: Copy, const N: usize> {
    pub data: [T; N],
}

pub type Mat<T, const W: usize, const H: usize> = Col<Col<T, W>, H>;

// impl<T: Copy, const N: usize> From<T> for Col<T, N> {
//     fn from(value: T) -> Self {
//         Col { data: [value; N] }
//     }
// }
//
// impl<T: Copy, const W: usize, const H: usize> From<T> for Mat<T, W, H> {
//     fn from(value: T) -> Self {
//         Col {
//             data: [value.into(); H],
//         }
//     }
// }

impl<T: Copy, const N: usize> From<[T; N]> for Col<T, N> {
    fn from(value: [T; N]) -> Self {
        Self { data: value }
    }
}

impl<T: Copy + One + Mul<T, Output = T>, const N: usize> One for Col<T, N>
where
    Col<T, N>: Splat<Col<T, N>>,
{
    fn one() -> Self {
        Self {
            data: [T::one(); N],
        }
    }
}

impl<T: Copy, const N: usize> IntoIterator for Col<T, N> {
    type Item = T;
    type IntoIter = <[T; N] as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl<T: Copy, const N: usize> Col<T, N> {
    pub fn map<F: FnMut(T) -> U, U: Copy>(self, f: F) -> Col<U, N> {
        Col::<U, N> {
            data: self.data.map(f),
        }
    }

    pub fn zip<F: FnMut(T, R) -> U, U: Copy, R: Copy>(self, r: Col<R, N>, mut f: F) -> Col<U, N> {
        Col::<U, N>::try_from_rows(zip(self.rows(), r.rows()).map(|(l, r)| f(l, r))).unwrap()
    }

    pub fn rows(self) -> <Self as IntoIterator>::IntoIter {
        self.into_iter()
    }

    pub fn try_from_rows(iter: impl IntoIterator<Item = T>) -> Option<Self> {
        Some(Self::from(
            <[T; N]>::try_from(iter.into_iter().collect::<Vec<_>>()).ok()?,
        ))
    }
}

pub trait Splat<T: Copy> {
    fn splat(self) -> T;
}

// impl<T: Copy + PartialOrd> Splat<T> for T {
//     fn splat(self) -> T {
//         self
//     }
// }

// vector to vector identity splat
impl<T: Copy, const N: usize> Splat<Col<T, N>> for Col<T, N> {
    fn splat(self) -> Col<T, N> {
        self
    }
}

// vector to vector identity splat
impl<T: Copy, const N: usize> Splat<Col<T, N>> for &Col<T, N> {
    fn splat(self) -> Col<T, N> {
        *self
    }
}

// scalar to vector splat
impl<T: Copy, const N: usize> Splat<Col<T, N>> for T
where
    T: PartialOrd,
{
    fn splat(self) -> Col<T, N> {
        Col::<T, N> { data: [self; N] }
    }
}

// scalar to vector splat
impl<T: Copy, const N: usize> Splat<Col<T, N>> for &T
where
    T: PartialOrd,
{
    fn splat(self) -> Col<T, N> {
        Col::<T, N> { data: [(*self); N] }
    }
}

// vector to matrix splat
impl<T: Copy + PartialOrd, const N: usize, const M: usize> Splat<Col<Col<T, N>, M>> for Col<T, M> {
    fn splat(self) -> Col<Col<T, N>, M> {
        self.map(Splat::splat)
    }
}

// vector to matrix splat
impl<T: Copy + PartialOrd, const N: usize, const M: usize> Splat<Col<Col<T, N>, M>> for &Col<T, M> {
    fn splat(self) -> Col<Col<T, N>, M> {
        self.map(Splat::splat)
    }
}

// scalar to matrix splat
impl<T: Copy, const N: usize, const M: usize> Splat<Col<Col<T, N>, M>> for T
where
    T: PartialOrd,
{
    fn splat(self) -> Col<Col<T, N>, M> {
        Col::<Col<T, N>, M> {
            data: [self.splat(); M],
        }
    }
}

// scalar to matrix splat
impl<T: Copy, const N: usize, const M: usize> Splat<Col<Col<T, N>, M>> for &T
where
    T: PartialOrd,
{
    fn splat(self) -> Col<Col<T, N>, M> {
        Col::<Col<T, N>, M> {
            data: [(*self).splat(); M],
        }
    }
}

//
// impl<T: Copy, const N: usize> Splat<Col<T, N> for Col<T, N> {
//     type Scalar = T;
//     fn splat(self) -> Col<T, N> {
//         self
//     }
// }
//
// impl<T: Copy, const N: usize> Splat<T, N> for &Col<T, N> {
//     type Scalar = T;
//     fn splat(self) -> Col<T, N> {
//         *self
//     }
// }
//
// impl<T: Copy, S: Copy const N : usize> Splat<T,N> for S where T : Splat<S>
//
// impl<T: Copy, const N: usize> Splat<T, N> for T
// where
//     Col<T, N>: One + Mul<T, Output = Col<T, N>>,
// {
//     type Scalar = T;
//     fn splat(self) -> Col<T, N> {
//         Col::<T, N>::one() * self
//     }
// }
//
// impl<T: Copy, const N: usize> Splat<T, N> for &T {
//     type Scalar = T;
//     fn splat(self) -> Col<T, N> {
//         Col::<T, N> { data: [*self; N] }
//     }
// }

// impl<T: Copy, const W: usize, const H: usize> Splat<Col<T, W>, H> for T {
//     type Scalar = T;
//     fn splat(self) -> Col<Col<T, W>, H> {
//         Col::<Col<T, W>, H> {
//             data: [self.splat(); H],
//         }
//     }
// }
//
// impl<T: Copy, const W: usize, const H: usize> Splat<Col<T, W>, H> for &T {
//     type Scalar = T;
//     fn splat(self) -> Col<Col<T, W>, H> {
//         Col::<Col<T, W>, H> {
//             data: [self.splat(); H],
//         }
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;
    use itertools::assert_equal;

    #[test]
    fn test_splat_scalar_to_col() {
        let a: Col<i32, 4> = 5.splat();
        assert_equal(a, Col::from([5, 5, 5, 5]))
    }

    #[test]
    fn test_splat_col_to_matrix() {
        let a = Col::from([1, 2, 3]);
        let b: Col<Col<_, 4>, 3> = a.splat();
        for n in 0..3 {
            assert_equal(b.data[n], Col::from([a.data[n]; 4]))
        }
    }

    #[test]
    fn test_splat_scalar_to_matrix() {
        let a: Col<Col<_, 4>, 3> = 5.splat();
        for n in 0..3 {
            assert_equal(a.data[n], Col::from([5; 4]));
        }
    }
}
