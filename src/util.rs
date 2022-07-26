use num_traits::{Num, NumOps, One, Zero};
use std::ops::Div;

pub fn checked_div<L: Num + Div<R, Output = T>, R: Num + Zero, T>(num: L, den: R) -> Option<T> {
    if den.is_zero() {
        return None;
    }
    return Some(num / den);
}

pub fn checked_inv<T: Num + Div<T, Output = T> + Zero + One>(den: T) -> Option<T> {
    return checked_div(T::one(), den);
}
