// borrowed from the auto_ops crate
#[doc(hidden)]
#[macro_export]
macro_rules! impl_matrix_op {
    (neg, $f:expr) => {
        $crate::_impl_op_m_internal_ex!(Neg, neg, $f);
    };
    (!, $f:expr) => {
        $crate::_impl_op_m_internal_ex!(Not, not, $f);
    };
    (+, $f:expr) => {
        $crate::_impl_op_mm_internal_ex!(Add, add, $f);
        $crate::_impl_opassign_mm_internal_ex!(Add, AddAssign, add_assign, $f);
    };
    (-, $f:expr) => {
        $crate::_impl_op_mm_internal_ex!(Sub, sub, $f);
        $crate::_impl_opassign_mm_internal_ex!(Sub, SubAssign, sub_assign, $f);
    };
    (*, $f:expr) => {
        $crate::_impl_op_mm_internal_ex!(Mul, mul, $f);
        $crate::_impl_op_ms_internal_ex!(Mul, mul, $f);
        $crate::_impl_opassign_mm_internal_ex!(Mul, MulAssign, mul_assign, $f);
        $crate::_impl_opassign_ms_internal_ex!(Mul, MulAssign, mul_assign, $f);
    };
    (/, $f:expr) => {
        $crate::_impl_op_mm_internal_ex!(Div, div, $f);
        $crate::_impl_op_ms_internal_ex!(Div, div, $f);
        $crate::_impl_opassign_mm_internal_ex!(Div, DivAssign, div_assign, $f);
        $crate::_impl_opassign_ms_internal_ex!(Div, DivAssign, div_assign, $f);
    };
    (%, $f:expr) => {
        $crate::_impl_op_mm_internal_ex!(Rem, rem, $f);
        $crate::_impl_op_ms_internal_ex!(Rem, rem, $f);
        $crate::_impl_opassign_mm_internal_ex!(Rem, RemAssign, rem_assign, $f);
        $crate::_impl_opassign_ms_internal_ex!(Rem, RemAssign, rem_assign, $f);
    };
    (&, $f:expr) => {
        $crate::_impl_op_mm_internal_ex!(BitAnd, bitand, $f);
        $crate::_impl_opassign_mm_internal_ex!(BitAnd, BitAndAssign, bitand_assign, $f);
    };
    (|, $f:expr) => {
        $crate::_impl_op_mm_internal_ex!(BitOr, bitor, $f);
        $crate::_impl_opassign_mm_internal_ex!(BitOr, BitOrAssign, bitor_assign, $f);
    };
    (^, $f:expr) => {
        $crate::_impl_op_mm_internal_ex!(BitXor, bitxor, $f);
        $crate::_impl_opassign_mm_internal_ex!(BitXor, BitXorAssign, bitxor_assign, $f);
    };
    (<<, $f:expr) => {
        $crate::_impl_op_ms_internal_ex!(Shl, shl, $f);
        $crate::_impl_opassign_mm_internal_ex!(Shl, ShlAssign, shl_assign, $f);
    };
    (>>, $f:expr) => {
        $crate::_impl_op_ms_internal_ex!(Shr, shr, $f);
        $crate::_impl_opassign_mm_internal_ex!(Shr, ShrAssign, shr_assign, $f);
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! impl_op_ms {
    (*, $f:expr) => {
        _impl_op_ms_internal!(Mul, mul, $f);
    };
    (/, $f:expr) => {
        _impl_op_ms_internal!(Div, div, $f);
    };
    (%, $f:expr) => {
        _impl_op_ms_internal!(Rem, rem, $f);
    };
    (<<, $f:expr) => {
        _impl_op_ms_internal!(Shl, shl, $f);
    };
    (>>, $f:expr) => {
        _impl_op_ms_internal!(Shr, shr, $d);
    };
}

#[macro_export]
macro_rules! _impl_op_m_internal_ex {
    ($ops_trait:ident, $ops_fn:ident, $f:expr) => {
        $crate::_impl_op_m_internal!($ops_trait, $ops_fn, Matrix<L,M,N>, Matrix<L,M,N>, $f);
        $crate::_impl_op_m_internal!($ops_trait, $ops_fn, &Matrix<L,M,N>, Matrix<L,M,N>, $f);
    }
}

#[macro_export]
macro_rules! _impl_op_mm_internal_ex {
    ($ops_trait:ident, $ops_fn:ident, $f:expr) => {
        $crate::_impl_op_mm_internal!($ops_trait, $ops_fn, Matrix<L,M,N>, Matrix<R,M,N>, Matrix<L,M,N>, $f);
        $crate::_impl_op_mm_internal!($ops_trait, $ops_fn, &Matrix<L,M,N>, Matrix<R,M,N>, Matrix<L,M,N>, $f);
        $crate::_impl_op_mm_internal!($ops_trait, $ops_fn, Matrix<L,M,N>, &Matrix<R,M,N>, Matrix<L,M,N>, $f);
        $crate::_impl_op_mm_internal!($ops_trait, $ops_fn, &Matrix<L,M,N>, &Matrix<R,M,N>, Matrix<L,M,N>, $f);
    }
}

#[macro_export]
macro_rules! _impl_opassign_mm_internal_ex {
    ($ops_super:ident, $ops_trait:ident, $ops_fn:ident, $f:expr) => {
        $crate::_impl_opassign_mm_internal!($ops_super, $ops_trait, $ops_fn, Matrix<L,M,N>, Matrix<R,M,N>, Matrix<L,M,N>, $f);
        $crate::_impl_opassign_mm_internal!($ops_super, $ops_trait, $ops_fn, Matrix<L,M,N>, &Matrix<R,M,N>, Matrix<L,M,N>, $f);
    }
}

#[macro_export]
macro_rules! _impl_op_ms_internal_ex {
    ($ops_trait:ident, $ops_fn:ident, $f:expr) => {
        $crate::_impl_op_ms_internal!($ops_trait, $ops_fn, Matrix<L,M,N>, R, Matrix<L,M,N>, $f);
        $crate::_impl_op_ms_internal!($ops_trait, $ops_fn, &Matrix<L,M,N>, R, Matrix<L,M,N>, $f);
    }
}

#[macro_export]
macro_rules! _impl_opassign_ms_internal_ex {
    ($ops_super:ident, $ops_trait:ident, $ops_fn:ident, $f:expr) => {
        $crate::_impl_opassign_ms_internal!($ops_super, $ops_trait, $ops_fn, Matrix<L,M,N>, R, Matrix<L,M,N>, $f);
    }
}

#[macro_export]
macro_rules! _impl_op_mm_internal {
    ($ops_trait:ident, $ops_fn:ident, $lhs:ty, $rhs:ty, $out:ty, $f:expr) => {
        impl<L, R, const M: usize, const N: usize> ::std::ops::$ops_trait<$rhs> for $lhs
        where
            L: ::std::ops::$ops_trait<R, Output = L>,
            L: Scalar,
            R: Scalar,
        {
            type Output = $out;

            fn $ops_fn(self, rhs_i: $rhs) -> Self::Output {
                let mut result = self.clone();
                let op = $f;
                for (l, r) in zip(result.elements_mut(), rhs_i.elements()) {
                    *l = op(*l, *r);
                }
                result
            }
        }
    };
}

#[macro_export]
macro_rules! _impl_opassign_mm_internal {
    ($ops_super:ident, $ops_trait:ident, $ops_fn:ident, $lhs:ty, $rhs:ty, $out:ty, $f:expr) => {
        impl<L, R, const M: usize, const N: usize> ::std::ops::$ops_trait<$rhs> for $lhs
        where
            L: ::std::ops::$ops_trait<R>,
            L: ::std::ops::$ops_super<R, Output = L>,
            L: Scalar,
            R: Scalar,
        {
            fn $ops_fn(&mut self, rhs_i: $rhs) {
                let op = $f;
                for (l, r) in zip(self.elements_mut(), rhs_i.elements()) {
                    *l = op(*l, *r);
                }
            }
        }
    };
}

#[macro_export]
macro_rules! _impl_op_m_internal {
    ($ops_trait:ident, $ops_fn:ident, $lhs:ty, $out:ty, $f:expr) => {
        impl<L, const M: usize, const N: usize> ::std::ops::$ops_trait for $lhs
        where
            L: ::std::ops::$ops_trait<Output = L>,
            L: Scalar,
        {
            type Output = $out;

            fn $ops_fn(self) -> Self::Output {
                let mut result = self.clone();
                let op = $f;
                for l in result.elements_mut() {
                    *l = op(*l);
                }
                result
            }
        }
    };
}

#[macro_export]
macro_rules! _impl_op_ms_internal {
    ($ops_trait:ident, $ops_fn:ident, $lhs:ty, $rhs:ty, $out:ty, $f:expr) => {
        impl<L, R, const M: usize, const N: usize> ::std::ops::$ops_trait<$rhs> for $lhs
        where
            L: ::std::ops::$ops_trait<R, Output = L>,
            L: Scalar,
            R: Scalar,
        {
            type Output = $out;

            fn $ops_fn(self, r: $rhs) -> Self::Output {
                let mut result = self.clone();
                let op = $f;
                for l in result.elements_mut() {
                    *l = op(*l, r);
                }
                result
            }
        }
    };
}

#[macro_export]
macro_rules! _impl_opassign_ms_internal {
    ($ops_super:ident, $ops_trait:ident, $ops_fn:ident, $lhs:ty, $rhs:ty, $out:ty, $f:expr) => {
        impl<L, R, const M: usize, const N: usize> ::std::ops::$ops_trait<$rhs> for $lhs
        where
            L: ::std::ops::$ops_trait<R>,
            L: ::std::ops::$ops_super<R, Output = L>,
            L: Scalar,
            R: Scalar,
        {
            fn $ops_fn(&mut self, r: $rhs) {
                let op = $f;
                for l in self.elements_mut() {
                    *l = op(*l, r);
                }
            }
        }
    };
}
