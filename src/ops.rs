// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use crate::{Col, Splat};
use num_traits::Num;
use std::ops::Mul;

// borrowed from the auto_ops crate
#[doc(hidden)]
macro_rules! impl_matrix_op {
    (neg) => {
        _impl_op_unary!(Neg::neg);
    };
    (!) => {
        _impl_op_unary!(Not::not);
    };
    (+) => {
        _impl_op_binary!(Add::add, AddAssign::add_assign);
    };
    (-) => {
        _impl_op_binary!(Sub::sub, SubAssign::sub_assign);
    };
    (*) => {
        _impl_op_binary!(Mul::mul, MulAssign::mul_assign);
    };
    (/) => {
        _impl_op_binary!(Div::div, DivAssign::div_assign);
    };
    (%) => {
        _impl_op_binary!(Rem::rem, RemAssign::rem_assign);
    };
    (&) => {
        _impl_op_binary!(BitAnd::bitand, BitAndAssign::bitand_assign);
    };
    (|) => {
        _impl_op_binary!(BitOr::bitor, BitOrAssign::bitor_assign);
    };
    (^) => {
        _impl_op_binary!(BitXor::bitxor, BitXorAssign::bitxor_assign);
    };
    (<<) => {
        _impl_op_binary!(Shl::shl, ShlAssign::shl_assign);
    };
    (>>) => {
        _impl_op_binary!(Shr::shr, ShrAssign::shr_assign);
    };
}

#[doc(hidden)]
macro_rules! _impl_op_unary {
    ($op_trait:ident::$op_fn:ident) => {
        impl<L, const N: usize> ::std::ops::$op_trait for Col<L, N>
        where
            L: ::std::ops::$op_trait<Output = L> + Copy,
        {
            type Output = Col<L, N>;

            #[inline(always)]
            fn $op_fn(self) -> Self::Output {
                let mut result = self.clone();
                // we arnt using iterators because they dont seem to always vectorize correctly
                for n in 0..N {
                    result.data[n] = self.data[n].$op_fn();
                }
                result
            }
        }

        impl<L, const N: usize> ::std::ops::$op_trait for &Col<L, N>
        where
            L: ::std::ops::$op_trait<Output = L> + Copy,
        {
            type Output = Col<L, N>;

            #[inline(always)]
            fn $op_fn(self) -> Self::Output {
                (*self).$op_fn()
            }
        }
    };
}

#[doc(hidden)]
macro_rules! _impl_op_binary {
    ($op_trait:ident::$op_fn:ident, $op_assign_trait:ident::$op_assign_fn:ident) => {
        impl<L, R, const N: usize> ::std::ops::$op_trait<R> for Col<L, N>
        where
            L: ::std::ops::$op_trait<L, Output = L> + Copy,
            R: Splat<Col<L, N>> + Copy,
        {
            type Output = Col<L, N>;

            #[inline(always)]
            fn $op_fn(self, other: R) -> Self::Output {
                let other_splat: Col<L, N> = other.splat();
                let mut result = self.clone();
                for n in 0..N {
                    result.data[n] = self.data[n].$op_fn(other_splat.data[n]);
                }
                result
            }
        }

        impl<L, R, const N: usize> ::std::ops::$op_trait<R> for &Col<L, N>
        where
            L: ::std::ops::$op_trait<L, Output = L> + Copy,
            R: Splat<Col<L, N>> + Copy,
        {
            type Output = Col<L, N>;

            #[inline(always)]
            fn $op_fn(self, other: R) -> Self::Output {
                (*self).$op_fn(other)
            }
        }

        impl<L, R, const N: usize> ::std::ops::$op_assign_trait<R> for Col<L, N>
        where
            L: ::std::ops::$op_assign_trait<L> + Copy,
            R: Splat<Col<L, N>> + Copy,
        {
            #[inline(always)]
            fn $op_assign_fn(&mut self, other: R) {
                let other_splat: Col<L, N> = other.splat();
                for n in 0..N {
                    self.data[n].$op_assign_fn(other_splat.data[n]);
                }
            }
        }
    };
}

impl_matrix_op!(neg);
impl_matrix_op!(!);
impl_matrix_op!(+);
impl_matrix_op!(-);
impl_matrix_op!(*);
impl_matrix_op!(/);
impl_matrix_op!(%);
impl_matrix_op!(&);
impl_matrix_op!(|);
impl_matrix_op!(^);
impl_matrix_op!(<<);
impl_matrix_op!(>>);

// impl<L: Copy, const N: usize> Mul<Col<L, N>> for Col<L, N>
// where
//     L: Mul<L, Output = L>,
// {
//     type Output = Col<L, N>;
//
//     fn mul(self, rhs: Col<L, N>) -> Self::Output {
//         self.zip(rhs, Mul::mul)
//     }
// }
//
// impl<L: Copy, const N: usize> Mul<L> for Col<L, N>
// where
//     L: Mul<L, Output = L>,
// {
//     type Output = Col<L, N>;
//
//     fn mul(self, rhs: Col<L, N>) -> Self::Output {
//         self.zip(rhs, Mul::mul)
//     }
// }
