// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::iter::zip;

use num_traits::{Float, NumCast, NumOps};

use vector_victor::Matrix;

pub trait Approx: PartialEq {
    fn approx(left: &Self, right: &Self) -> bool {
        left == right
    }
}

macro_rules! multi_impl { ($name:ident for $($t:ty),*) => ($( impl $name for $t {} )*) }
multi_impl!(Approx for i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize, bool);

impl Approx for f32 {
    fn approx(left: &f32, right: &f32) -> bool {
        f32::abs(left - right) <= f32::epsilon()
    }
}

impl Approx for f64 {
    fn approx(left: &f64, right: &f64) -> bool {
        f64::abs(left - right) <= f32::epsilon() as f64
    }
}

impl<T: Copy + Approx, const M: usize, const N: usize> Approx for Matrix<T, M, N> {
    fn approx(left: &Self, right: &Self) -> bool {
        zip(left.elements(), right.elements()).all(|(l, r)| T::approx(l, r))
    }
}

pub fn approx<T: Approx>(left: &T, right: &T) -> bool {
    T::approx(left, right)
}

#[macro_export]
macro_rules! assert_approx {
    ($left:expr, $right:expr $(,)?) => {
        match (&$left, &$right) {
            (_left_val, _right_val) => {
                assert_approx!($left, $right, "Difference is less than epsilon")
            }
        }
    };
    ($left:expr, $right:expr, $($arg:tt)+) => {
        match (&$left, &$right) {
            (left_val, right_val) => {


                if !common::approx(left_val, right_val){
                    assert_eq!(left_val, right_val, $($arg)+) // done this way to get nice errors
                }
            }
        }
    };
}

pub fn step<T, U>(start: U, step: U) -> impl Iterator<Item = T>
where
    T: NumCast,
    U: NumOps + NumCast + Copy,
{
    (0usize..).map_while(move |i| T::from((U::from(i)? + start) * step))
}
