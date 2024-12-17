// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

pub mod legacy;
mod ops;

extern crate core;

use itertools::Itertools;
use num_traits::{Bounded, One, Zero};
use std::cmp::min;
use std::fmt::Debug;
use std::iter::{zip, Flatten};
use std::ops::{Add, Index, IndexMut};

pub use legacy::{Matrix, Vector};

#[derive(Copy, Clone)]
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

pub trait Splat<T: Copy, const N: usize> {
    type Scalar;
    fn splat(self) -> Col<T, N>;
}

impl<T: Copy, const N: usize> Splat<T, N> for Col<T, N> {
    type Scalar = T;
    fn splat(self) -> Col<T, N> {
        self
    }
}

impl<T: Copy, const N: usize> Splat<T, N> for &Col<T, N> {
    type Scalar = T;
    fn splat(self) -> Col<T, N> {
        *self
    }
}

impl<T: Copy, const N: usize> Splat<T, N> for T {
    type Scalar = T;
    fn splat(self) -> Col<T, N> {
        Col::<T, N> { data: [self; N] }
    }
}

impl<T: Copy, const N: usize> Splat<T, N> for &T {
    type Scalar = T;
    fn splat(self) -> Col<T, N> {
        Col::<T, N> { data: [*self; N] }
    }
}

impl<T: Copy, const W: usize, const H: usize> Splat<Col<T, W>, H> for T {
    type Scalar = T;
    fn splat(self) -> Col<Col<T, W>, H> {
        Col::<Col<T, W>, H> {
            data: [self.splat(); H],
        }
    }
}

impl<T: Copy, const W: usize, const H: usize> Splat<Col<T, W>, H> for &T {
    type Scalar = T;
    fn splat(self) -> Col<Col<T, W>, H> {
        Col::<Col<T, W>, H> {
            data: [self.splat(); H],
        }
    }
}

trait SplatFrom<T> {
    fn splat_from(other: T) -> Self;
}

impl<T: Copy, const N: usize> SplatFrom<Col<T, N>> for Col<T, N> {
    fn splat_from(other: Col<T, N>) -> Self {
        other
    }
}

impl<T: Copy, const N: usize> SplatFrom<T> for Col<T, N> {
    fn splat_from(other: T) -> Self {
        Col::<T, N> { data: [other; N] }
    }
}

impl<T: Copy + Default, const N: usize> Default for Col<T, N> {
    fn default() -> Self {
        T::default().splat()
    }
}
