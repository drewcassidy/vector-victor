// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use crate::Matrix;
use num_traits::Num;

// borrowed from the auto_ops crate
#[doc(hidden)]
macro_rules! impl_matrix_op {
    (neg) => {
        _impl_op_unary_ex!(Neg::neg);
    };
    (!) => {
        _impl_op_unary_ex!(Not::not);
    };
    (+) => {
        _impl_op_binary_ex!(Add::add, AddAssign::add_assign);
    };
    (-) => {
        _impl_op_binary_ex!(Sub::sub, SubAssign::sub_assign);
    };
    (*) => {
        _impl_op_binary_ex!(Mul::mul, MulAssign::mul_assign);
    };
    (/) => {
        _impl_op_binary_ex!(Div::div, DivAssign::div_assign);
    };
    (%) => {
        _impl_op_binary_ex!(Rem::rem, RemAssign::rem_assign);
    };
    (&) => {
        _impl_op_binary_ex!(BitAnd::bitand, BitAndAssign::bitand_assign);
    };
    (|) => {
        _impl_op_binary_ex!(BitOr::bitor, BitOrAssign::bitor_assign);
    };
    (^) => {
        _impl_op_binary_ex!(BitXor::bitxor, BitXorAssign::bitxor_assign);
    };
    (<<) => {
        _impl_op_binary_ex!(Shl::shl, ShlAssign::shl_assign);
    };
    (>>) => {
        _impl_op_binary_ex!(Shr::shr, ShrAssign::shr_assign);
    };
}

#[doc(hidden)]
macro_rules! _impl_op_unary_ex {
    ($op_trait:ident::$op_fn:ident) => {
        _impl_op_m_internal!($op_trait, $op_fn, Matrix<L,M,N>, Matrix<L,M,N>);
        _impl_op_m_internal!($op_trait, $op_fn, &Matrix<L,M,N>, Matrix<L,M,N>);
    }
}

#[doc(hidden)]
macro_rules! _impl_op_binary_ex {
    ($op_trait:ident::$op_fn:ident, $op_assign_trait:ident::$op_assign_fn:ident) => {
        _impl_op_mm_internal!($op_trait, $op_fn, Matrix<L,M,N>, Matrix<R,M,N>, Matrix<L,M,N>);
        _impl_op_mm_internal!($op_trait, $op_fn, &Matrix<L,M,N>, Matrix<R,M,N>, Matrix<L,M,N>);
        _impl_op_mm_internal!($op_trait, $op_fn, Matrix<L,M,N>, &Matrix<R,M,N>, Matrix<L,M,N>);
        _impl_op_mm_internal!($op_trait, $op_fn, &Matrix<L,M,N>, &Matrix<R,M,N>, Matrix<L,M,N>);

        _impl_op_ms_internal!($op_trait, $op_fn, Matrix<L,M,N>, R, Matrix<L,M,N>);
        _impl_op_ms_internal!($op_trait, $op_fn, &Matrix<L,M,N>, R, Matrix<L,M,N>);

        _impl_opassign_mm_internal!($op_assign_trait, $op_assign_fn, Matrix<L,M,N>, Matrix<R,M,N>, Matrix<L,M,N>);
        _impl_opassign_mm_internal!($op_assign_trait, $op_assign_fn, Matrix<L,M,N>, &Matrix<R,M,N>, Matrix<L,M,N>);

        _impl_opassign_ms_internal!($op_assign_trait, $op_assign_fn, Matrix<L,M,N>, R, Matrix<L,M,N>);

    }
}

#[doc(hidden)]
macro_rules! _impl_op_m_internal {
    ($op_trait:ident, $op_fn:ident, $lhs:ty, $out:ty) => {
        impl<L, const M: usize, const N: usize> ::std::ops::$op_trait for $lhs
        where
            L: ::std::ops::$op_trait<Output = L> + Copy,
        {
            type Output = $out;

            #[inline(always)]
            fn $op_fn(self) -> Self::Output {
                let mut result = self.clone();
                // we arnt using iterators because they dont seem to always vectorize correctly
                for m in 0..M {
                    for n in 0..N {
                        result.data[m][n] = self.data[m][n].$op_fn();
                    }
                }
                result
            }
        }
    };
}

#[doc(hidden)]
macro_rules! _impl_op_mm_internal {
    ($op_trait:ident, $op_fn:ident, $lhs:ty, $rhs:ty, $out:ty) => {
        impl<L, R, const M: usize, const N: usize> ::std::ops::$op_trait<$rhs> for $lhs
        where
            L: ::std::ops::$op_trait<R, Output = L> + Copy,
            R: Copy,
        {
            type Output = $out;

            #[inline(always)]
            fn $op_fn(self, other: $rhs) -> Self::Output {
                let mut result = self.clone();
                for m in 0..M {
                    for n in 0..N {
                        result.data[m][n] = self.data[m][n].$op_fn(other.data[m][n]);
                    }
                }
                result
            }
        }
    };
}

#[doc(hidden)]
macro_rules! _impl_opassign_mm_internal {
    ($op_trait:ident, $op_fn:ident, $lhs:ty, $rhs:ty, $out:ty) => {
        impl<L, R, const M: usize, const N: usize> ::std::ops::$op_trait<$rhs> for $lhs
        where
            L: ::std::ops::$op_trait<R> + Copy,
            R: Copy,
        {
            #[inline(always)]
            fn $op_fn(&mut self, other: $rhs) {
                for m in 0..M {
                    for n in 0..N {
                        self.data[m][n].$op_fn(other.data[m][n]);
                    }
                }
            }
        }
    };
}

#[doc(hidden)]
macro_rules! _impl_op_ms_internal {
    ($op_trait:ident, $op_fn:ident, $lhs:ty, $rhs:ty, $out:ty) => {
        impl<L, R, const M: usize, const N: usize> ::std::ops::$op_trait<$rhs> for $lhs
        where
            L: ::std::ops::$op_trait<R, Output = L> + Copy,
            R: Copy + Num,
        {
            type Output = $out;

            #[inline(always)]
            fn $op_fn(self, other: $rhs) -> Self::Output {
                let mut result = self.clone();
                for m in 0..M {
                    for n in 0..N {
                        result.data[m][n] = self.data[m][n].$op_fn(other);
                    }
                }
                result
            }
        }
    };
}

#[doc(hidden)]
macro_rules! _impl_opassign_ms_internal {
    ($op_trait:ident, $op_fn:ident, $lhs:ty, $rhs:ty, $out:ty) => {
        impl<L, R, const M: usize, const N: usize> ::std::ops::$op_trait<$rhs> for $lhs
        where
            L: ::std::ops::$op_trait<R> + Copy,
            R: Copy + Num,
        {
            #[inline(always)]
            fn $op_fn(&mut self, r: $rhs) {
                for m in 0..M {
                    for n in 0..N {
                        self.data[m][n].$op_fn(r);
                    }
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
