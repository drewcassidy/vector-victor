use crate::matrix::Matrix;
use num_traits::Num;

// borrowed from the auto_ops crate
#[doc(hidden)]
macro_rules! impl_matrix_op {
    (neg) => {
        _impl_op_m_internal_ex!(Neg, neg);
    };
    (!) => {
        _impl_op_m_internal_ex!(Not, not);
    };
    (+) => {
        _impl_op_mm_internal_ex!(Add, add);
        _impl_opassign_mm_internal_ex!(AddAssign, add_assign);
    };
    (-) => {
        _impl_op_mm_internal_ex!(Sub, sub);
        _impl_opassign_mm_internal_ex!(SubAssign, sub_assign);
    };
    (*) => {
        _impl_op_mm_internal_ex!(Mul, mul);
        _impl_op_ms_internal_ex!(Mul, mul);
        _impl_opassign_mm_internal_ex!(MulAssign, mul_assign);
        _impl_opassign_ms_internal_ex!(MulAssign, mul_assign);
    };
    (/) => {
        _impl_op_mm_internal_ex!(Div, div);
        _impl_op_ms_internal_ex!(Div, div);
        _impl_opassign_mm_internal_ex!(DivAssign, div_assign);
        _impl_opassign_ms_internal_ex!(DivAssign, div_assign);
    };
    (%) => {
        _impl_op_mm_internal_ex!(Rem, rem);
        _impl_op_ms_internal_ex!(Rem, rem);
        _impl_opassign_mm_internal_ex!(RemAssign, rem_assign);
        _impl_opassign_ms_internal_ex!(RemAssign, rem_assign);
    };
    (&) => {
        _impl_op_mm_internal_ex!(BitAnd, bitand);
        _impl_opassign_mm_internal_ex!(BitAndAssign, bitand_assign);
    };
    (|) => {
        _impl_op_mm_internal_ex!(BitOr, bitor);
        _impl_opassign_mm_internal_ex!(BitOrAssign, bitor_assign);
    };
    (^) => {
        _impl_op_mm_internal_ex!(BitXor, bitxor);
        _impl_opassign_mm_internal_ex!(BitXorAssign, bitxor_assign);
    };
    (<<) => {
        _impl_op_ms_internal_ex!(Shl, shl);
        _impl_opassign_ms_internal_ex!(ShlAssign, shl_assign);
    };
    (>>) => {
        _impl_op_ms_internal_ex!(Shr, shr);
        _impl_opassign_ms_internal_ex!(ShrAssign, shr_assign);
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! _impl_op_m_internal_ex {
    ($ops_trait:ident, $ops_fn:ident) => {
        _impl_op_m_internal!($ops_trait, $ops_fn, Matrix<L,M,N>, Matrix<L,M,N>);
        _impl_op_m_internal!($ops_trait, $ops_fn, &Matrix<L,M,N>, Matrix<L,M,N>);
    }
}

#[doc(hidden)]
#[macro_export]
macro_rules! _impl_op_mm_internal_ex {
    ($ops_trait:ident, $ops_fn:ident) => {
        _impl_op_mm_internal!($ops_trait, $ops_fn, Matrix<L,M,N>, Matrix<R,M,N>, Matrix<L,M,N>);
        _impl_op_mm_internal!($ops_trait, $ops_fn, &Matrix<L,M,N>, Matrix<R,M,N>, Matrix<L,M,N>);
        _impl_op_mm_internal!($ops_trait, $ops_fn, Matrix<L,M,N>, &Matrix<R,M,N>, Matrix<L,M,N>);
        _impl_op_mm_internal!($ops_trait, $ops_fn, &Matrix<L,M,N>, &Matrix<R,M,N>, Matrix<L,M,N>);
    }
}

#[doc(hidden)]
#[macro_export]
macro_rules! _impl_opassign_mm_internal_ex {
    ($ops_trait:ident, $ops_fn:ident) => {
        _impl_opassign_mm_internal!($ops_trait, $ops_fn, Matrix<L,M,N>, Matrix<R,M,N>, Matrix<L,M,N>);
        _impl_opassign_mm_internal!($ops_trait, $ops_fn, Matrix<L,M,N>, &Matrix<R,M,N>, Matrix<L,M,N>);
    }
}

#[doc(hidden)]
macro_rules! _impl_op_ms_internal_ex {
    ($ops_trait:ident, $ops_fn:ident) => {
        _impl_op_ms_internal!($ops_trait, $ops_fn, Matrix<L,M,N>, R, Matrix<L,M,N>);
        _impl_op_ms_internal!($ops_trait, $ops_fn, &Matrix<L,M,N>, R, Matrix<L,M,N>);
    }
}

#[doc(hidden)]
macro_rules! _impl_opassign_ms_internal_ex {
    ($ops_trait:ident, $ops_fn:ident) => {
        _impl_opassign_ms_internal!($ops_trait, $ops_fn, Matrix<L,M,N>, R, Matrix<L,M,N>);
    }
}

#[doc(hidden)]
macro_rules! _impl_op_m_internal {
    ($ops_trait:ident, $ops_fn:ident, $lhs:ty, $out:ty) => {
        impl<L, const M: usize, const N: usize> ::std::ops::$ops_trait for $lhs
        where
            L: ::std::ops::$ops_trait<Output = L> + Copy,
        {
            type Output = $out;

            #[inline(always)]
            fn $ops_fn(self) -> Self::Output {
                let mut result = self.clone();
                for m in 0..M {
                    for n in 0..N {
                        result.data[m][n] = self.data[m][n].$ops_fn();
                    }
                }
                result
            }
        }
    };
}

#[doc(hidden)]
macro_rules! _impl_op_mm_internal {
    ($ops_trait:ident, $ops_fn:ident, $lhs:ty, $rhs:ty, $out:ty) => {
        impl<L, R, const M: usize, const N: usize> ::std::ops::$ops_trait<$rhs> for $lhs
        where
            L: ::std::ops::$ops_trait<R, Output = L> + Copy,
            R: Copy,
        {
            type Output = $out;

            #[inline(always)]
            fn $ops_fn(self, other: $rhs) -> Self::Output {
                let mut result = self.clone();
                for m in 0..M {
                    for n in 0..N {
                        result.data[m][n] = self.data[m][n].$ops_fn(other.data[m][n]);
                    }
                }
                result
            }
        }
    };
}

#[doc(hidden)]
macro_rules! _impl_opassign_mm_internal {
    ($ops_trait:ident, $ops_fn:ident, $lhs:ty, $rhs:ty, $out:ty) => {
        impl<L, R, const M: usize, const N: usize> ::std::ops::$ops_trait<$rhs> for $lhs
        where
            L: ::std::ops::$ops_trait<R> + Copy,
            R: Copy,
        {
            #[inline(always)]
            fn $ops_fn(&mut self, other: $rhs) {
                for m in 0..M {
                    for n in 0..N {
                        self.data[m][n].$ops_fn(other.data[m][n]);
                    }
                }
            }
        }
    };
}

#[doc(hidden)]
macro_rules! _impl_op_ms_internal {
    ($ops_trait:ident, $ops_fn:ident, $lhs:ty, $rhs:ty, $out:ty) => {
        impl<L, R, const M: usize, const N: usize> ::std::ops::$ops_trait<$rhs> for $lhs
        where
            L: ::std::ops::$ops_trait<R, Output = L> + Copy,
            R: Copy + Num,
        {
            type Output = $out;

            #[inline(always)]
            fn $ops_fn(self, other: $rhs) -> Self::Output {
                let mut result = self.clone();
                for m in 0..M {
                    for n in 0..N {
                        result.data[m][n] = self.data[m][n].$ops_fn(other);
                    }
                }
                result
            }
        }
    };
}

#[doc(hidden)]
macro_rules! _impl_opassign_ms_internal {
    ($ops_trait:ident, $ops_fn:ident, $lhs:ty, $rhs:ty, $out:ty) => {
        impl<L, R, const M: usize, const N: usize> ::std::ops::$ops_trait<$rhs> for $lhs
        where
            L: ::std::ops::$ops_trait<R> + Copy,
            R: Copy + Num,
        {
            #[inline(always)]
            fn $ops_fn(&mut self, r: $rhs) {
                for m in 0..M {
                    for n in 0..N {
                        self.data[m][n].$ops_fn(r);
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
