// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

pub mod legacy;
mod math;
mod ops;

extern crate core;

pub use legacy::{Matrix, Vector};
use num_traits::{Num, One, Zero};
use std::fmt::Debug;
use std::iter::zip;
use std::ops::{Add, Mul};

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct Col<T: Copy, const N: usize> {
    pub data: [T; N],
}

pub type Mat<T, const H: usize, const W: usize> = Col<Col<T, W>, H>;

impl<T: Copy, const N: usize> From<[T; N]> for Col<T, N> {
    fn from(value: [T; N]) -> Self {
        Self { data: value }
    }
}

impl<T: Copy + Default, const N: usize> Default for Col<T, N> {
    fn default() -> Self {
        Self::from([T::default(); N])
    }
}

impl<T: Copy + One + Add<T, Output = T>, const N: usize> One for Col<T, N>
where
    Col<T, N>: Mul<Col<T, N>, Output = Col<T, N>>,
{
    fn one() -> Self {
        Self::from([T::one(); N])
    }
}

impl<T: Copy + Zero, const N: usize> Zero for Col<T, N>
where
    Col<T, N>: Add<Col<T, N>, Output = Col<T, N>>,
{
    fn zero() -> Self {
        Self::from([T::zero(); N])
    }

    fn is_zero(&self) -> bool {
        self.rows().all(|r| r.is_zero())
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

pub trait Scalar {}

impl<T> Scalar for T where T: Num {}

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
impl<T: Scalar + Copy, const N: usize> Splat<Col<T, N>> for T {
    fn splat(self) -> Col<T, N> {
        Col::<T, N> { data: [self; N] }
    }
}

/// Scalar to Vector splat (Reference)
impl<T: Scalar + Copy, const N: usize> Splat<Col<T, N>> for &T {
    fn splat(self) -> Col<T, N> {
        Col::<T, N> { data: [(*self); N] }
    }
}

/// Vector to Matrix splat
impl<T: Scalar + Copy, const N: usize, const M: usize> Splat<Col<Col<T, N>, M>> for Col<T, M> {
    fn splat(self) -> Col<Col<T, N>, M> {
        self.map(Splat::splat)
    }
}

/// Vector to Matrix splat (Reference)
impl<T: Scalar + Copy, const N: usize, const M: usize> Splat<Col<Col<T, N>, M>> for &Col<T, M> {
    fn splat(self) -> Col<Col<T, N>, M> {
        self.map(Splat::splat)
    }
}

/// Scalar to Matrix splat
impl<T: Scalar + Copy, const N: usize, const M: usize> Splat<Col<Col<T, N>, M>> for T {
    fn splat(self) -> Col<Col<T, N>, M> {
        Col::<Col<T, N>, M> {
            data: [self.splat(); M],
        }
    }
}

/// Scalar to Matrix splat (Reference)
impl<T: Scalar + Copy, const N: usize, const M: usize> Splat<Col<Col<T, N>, M>> for &T {
    fn splat(self) -> Col<Col<T, N>, M> {
        Col::<Col<T, N>, M> {
            data: [(*self).splat(); M],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
