// borrowed from the auto_ops crate
#[doc(hidden)]
#[macro_export]
macro_rules! impl_matrix_op {
    (neg) => {
        $crate::_impl_op_m_internal_ex!(Neg, neg);
    };
    (!) => {
        $crate::_impl_op_m_internal_ex!(Not, not);
    };
    (+) => {
        $crate::_impl_op_mm_internal_ex!(Add, add);
        $crate::_impl_opassign_mm_internal_ex!(AddAssign, add_assign);
    };
    (-) => {
        $crate::_impl_op_mm_internal_ex!(Sub, sub);
        $crate::_impl_opassign_mm_internal_ex!(SubAssign, sub_assign);
    };
    (*) => {
        $crate::_impl_op_mm_internal_ex!(Mul, mul);
        $crate::_impl_op_ms_internal_ex!(Mul, mul);
        $crate::_impl_opassign_mm_internal_ex!(MulAssign, mul_assign);
        $crate::_impl_opassign_ms_internal_ex!(MulAssign, mul_assign);
    };
    (/) => {
        $crate::_impl_op_mm_internal_ex!(Div, div);
        $crate::_impl_op_ms_internal_ex!(Div, div);
        $crate::_impl_opassign_mm_internal_ex!(DivAssign, div_assign);
        $crate::_impl_opassign_ms_internal_ex!(DivAssign, div_assign);
    };
    (%) => {
        $crate::_impl_op_mm_internal_ex!(Rem, rem);
        $crate::_impl_op_ms_internal_ex!(Rem, rem);
        $crate::_impl_opassign_mm_internal_ex!(RemAssign, rem_assign);
        $crate::_impl_opassign_ms_internal_ex!(RemAssign, rem_assign);
    };
    (&) => {
        $crate::_impl_op_mm_internal_ex!(BitAnd, bitand);
        $crate::_impl_opassign_mm_internal_ex!(BitAndAssign, bitand_assign);
    };
    (|) => {
        $crate::_impl_op_mm_internal_ex!(BitOr, bitor);
        $crate::_impl_opassign_mm_internal_ex!(BitOrAssign, bitor_assign);
    };
    (^) => {
        $crate::_impl_op_mm_internal_ex!(BitXor, bitxor);
        $crate::_impl_opassign_mm_internal_ex!(BitXorAssign, bitxor_assign);
    };
    (<<) => {
        $crate::_impl_op_ms_internal_ex!(Shl, shl);
        $crate::_impl_opassign_ms_internal_ex!(ShlAssign, shl_assign);
    };
    (>>) => {
        $crate::_impl_op_ms_internal_ex!(Shr, shr);
        $crate::_impl_opassign_ms_internal_ex!(ShrAssign, shr_assign);
    };
}

#[macro_export]
macro_rules! _impl_op_m_internal_ex {
    ($ops_trait:ident, $ops_fn:ident) => {
        $crate::_impl_op_m_internal!($ops_trait, $ops_fn, Matrix<L,M,N>, Matrix<L,M,N>);
        $crate::_impl_op_m_internal!($ops_trait, $ops_fn, &Matrix<L,M,N>, Matrix<L,M,N>);
    }
}

#[macro_export]
macro_rules! _impl_op_mm_internal_ex {
    ($ops_trait:ident, $ops_fn:ident) => {
        $crate::_impl_op_mm_internal!($ops_trait, $ops_fn, Matrix<L,M,N>, Matrix<R,M,N>, Matrix<L,M,N>);
        $crate::_impl_op_mm_internal!($ops_trait, $ops_fn, &Matrix<L,M,N>, Matrix<R,M,N>, Matrix<L,M,N>);
        $crate::_impl_op_mm_internal!($ops_trait, $ops_fn, Matrix<L,M,N>, &Matrix<R,M,N>, Matrix<L,M,N>);
        $crate::_impl_op_mm_internal!($ops_trait, $ops_fn, &Matrix<L,M,N>, &Matrix<R,M,N>, Matrix<L,M,N>);
    }
}

#[macro_export]
macro_rules! _impl_opassign_mm_internal_ex {
    ($ops_trait:ident, $ops_fn:ident) => {
        $crate::_impl_opassign_mm_internal!($ops_trait, $ops_fn, Matrix<L,M,N>, Matrix<R,M,N>, Matrix<L,M,N>);
        $crate::_impl_opassign_mm_internal!($ops_trait, $ops_fn, Matrix<L,M,N>, &Matrix<R,M,N>, Matrix<L,M,N>);
    }
}

#[macro_export]
macro_rules! _impl_op_ms_internal_ex {
    ($ops_trait:ident, $ops_fn:ident) => {
        $crate::_impl_op_ms_internal!($ops_trait, $ops_fn, Matrix<L,M,N>, R, Matrix<L,M,N>);
        $crate::_impl_op_ms_internal!($ops_trait, $ops_fn, &Matrix<L,M,N>, R, Matrix<L,M,N>);
    }
}

#[macro_export]
macro_rules! _impl_opassign_ms_internal_ex {
    ($ops_trait:ident, $ops_fn:ident) => {
        $crate::_impl_opassign_ms_internal!($ops_trait, $ops_fn, Matrix<L,M,N>, R, Matrix<L,M,N>);
    }
}

#[macro_export]
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

#[macro_export]
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

#[macro_export]
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

#[macro_export]
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

#[macro_export]
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
