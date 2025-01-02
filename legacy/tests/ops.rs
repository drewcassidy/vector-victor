// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#[macro_use]
mod common;

use crate::common::{step, Approx};
use generic_parameterize::parameterize;
use num_traits::{NumAssign, NumCast};
use std::fmt::Debug;
use std::iter::zip;
use std::ops::*;
use vector_victor::Matrix;

#[parameterize(S = (i32, f32), M = [1,4], N = [1,4], fmt = "{fn}_{S}_{M}x{N}")]
#[test]
fn neg<
    S: Copy + NumCast + NumAssign + Approx + Default + Debug + Neg<Output = S>,
    const M: usize,
    const N: usize,
>() {
    let a: Matrix<S, M, N> = step(-2, 2).collect();
    let expected: Matrix<S, M, N> = a.elements().map(|&a| -a).collect();

    assert_approx!(-a, expected, "Incorrect value for negation");
}

#[parameterize(S = (i32, u32), M = [1,4], N = [1,4], fmt = "{fn}_{S}_{M}x{N}")]
#[test]
fn not<
    S: Copy + NumCast + NumAssign + Approx + Default + Debug + Not<Output = S>,
    const M: usize,
    const N: usize,
>() {
    let a: Matrix<S, M, N> = step(-2, 2).collect();
    let expected: Matrix<S, M, N> = a.elements().map(|&a| !a).collect();

    assert_approx!(!a, expected, "Incorrect value for inversion");
}

#[parameterize(M = [1,4], N = [1,4], fmt="{fn}_{M}x{N}")]
#[test]
fn not_bool<const M: usize, const N: usize>() {
    let a: Matrix<bool, M, N> = [true, true, false].iter().cycle().copied().collect();
    let expected: Matrix<bool, M, N> = [false, false, true].iter().cycle().copied().collect();

    assert_approx!(!a, expected, "Incorrect value for inversion");
}

macro_rules! test_op {
    {$op_trait:ident::$op_fn:ident, $op_assign_trait:ident::$op_assign_fn:ident,
     $op_name:literal, $t:ty} => {
        #[parameterize(S = $t, M = [1,4], N = [1,4], fmt="{fn}_{S}_{M}x{N}")]
        #[test]
        fn $op_fn<
            S: Copy + NumCast + NumAssign + Approx + Default + Debug
                + $op_trait<S, Output=S>,
            const M: usize,
            const N: usize,
        >() {
            let a: Matrix<S, M, N> = step(2, 3).collect();
            let b: Matrix<S, M, N> = step(1, 2).collect();
            let expected: Matrix<S, M, N> = zip(a, b).map(|(aa, bb)| $op_trait::$op_fn(aa, bb)).collect();

            assert_approx!($op_trait::$op_fn(a, b), expected, "Incorrect value for {}", $op_name);
            assert_approx!($op_trait::$op_fn(a, &b), expected, "Incorrect value for {}", $op_name);
            assert_approx!($op_trait::$op_fn(&a, b), expected, "Incorrect value for {}", $op_name);
            assert_approx!($op_trait::$op_fn(&a, &b), expected, "Incorrect value for {}", $op_name);

            let s: S = S::from(2).unwrap();
            let expected: Matrix<S, M, N> = a.elements().map(|&aa| $op_trait::$op_fn(aa, s)).collect();

            assert_approx!($op_trait::$op_fn(a, s), expected, "Incorrect value for {} by scalar", $op_name);
            assert_approx!($op_trait::$op_fn(&a, s), expected, "Incorrect value for {} by scalar", $op_name);

        }

        #[parameterize(S = $t, M = [1,4], N = [1,4])]
        #[test]
        fn $op_assign_fn<
            S: Copy + NumCast + NumAssign + Approx + Default + Debug
                + $op_trait<S, Output=S> + $op_assign_trait<S>,
            const M: usize,
            const N: usize,
        >() {
            let a: Matrix<S, M, N> = step(2, 3).collect();
            let b: Matrix<S, M, N> = step(1, 2).collect();
            let expected: Matrix<S, M, N> = zip(a, b).map(|(aa, bb)| $op_trait::$op_fn(aa, bb)).collect();

            let mut c = a;
            $op_assign_trait::$op_assign_fn(&mut c, b);
            assert_approx!(c, expected, "Incorrect value for {}-assignment", $op_name);

            let mut c = a;
            $op_assign_trait::$op_assign_fn(&mut c, &b);
            assert_approx!(c, expected, "Incorrect value for {}-assignment", $op_name);

            let s: S = S::from(2).unwrap();
            let expected: Matrix<S, M, N> = a.elements().map(|&aa| $op_trait::$op_fn(aa, s)).collect();

            let mut c = a;
            $op_assign_trait::$op_assign_fn(&mut c, s);
            assert_approx!(c, expected, "Incorrect value for {}-assignment by scalar", $op_name);
        }
    };
}

test_op!(Add::add, AddAssign::add_assign, "addition", (i32, u32, f32));

test_op!(
    Sub::sub,
    SubAssign::sub_assign,
    "subtraction",
    (i32, u32, f32)
);

test_op!(
    Mul::mul,
    MulAssign::mul_assign,
    "multiplication",
    (i32, u32, f32)
);
test_op!(Div::div, DivAssign::div_assign, "division", (i32, u32, f32));

test_op!(
    Rem::rem,
    RemAssign::rem_assign,
    "remainder",
    (i32, u32, f32)
);

test_op!(BitOr::bitor, BitOrAssign::bitor_assign, "or", (i32, u32));

test_op!(
    BitAnd::bitand,
    BitAndAssign::bitand_assign,
    "and",
    (i32, u32)
);

test_op!(
    BitXor::bitxor,
    BitXorAssign::bitxor_assign,
    "xor",
    (i32, u32)
);

test_op!(Shl::shl, ShlAssign::shl_assign, "shift-left", (usize,));
test_op!(Shr::shr, ShrAssign::shr_assign, "shift-right", (usize,));
