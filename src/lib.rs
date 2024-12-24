// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

pub mod legacy;
mod math;
mod ops;
pub mod splat;

extern crate core;

use num_traits::{Num, One, Zero};
use std::fmt::Debug;
use std::iter::zip;
use std::ops::{Add, Mul};

pub use legacy::{Matrix, Vector};
pub use splat::{Scalar, Splat};

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
