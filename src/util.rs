// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use num_traits::{Num, One, Zero};
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
