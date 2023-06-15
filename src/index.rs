// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Helper trait for ergonomic matrix subscripting

use std::fmt::Debug;

/** Trait implemented by types that can be used as a matrix index

There are currently two implementations:
[`usize`](#impl-Index2D-for-usize) and [`(usize,usize)`](#impl-Index2D-for-(usize,+usize))

# Examples
Indexing by a `usize` indexes starting at the first element and
increments linearly in row-major order. This is especially useful for column vectors.

```
# use vector_victor::{Matrix, Vector};
let m = Matrix::mat([[1,2,3],[4,5,6],[7,8,9]]);
assert_eq!(m[0], 1);
assert_eq!(m[4], 5);
assert_eq!(m[7], 8);

let v = Vector::vec([4,8,15,16,23,42]);
assert_eq!(v[2], 15); // just like a std::vec
```

Indexing by a `(usize,usize)` indexes by row and column
```
# use vector_victor::{Matrix, Vector};
let m = Matrix::mat([[1,2,3],[4,5,6],[7,8,9]]);
assert_eq!(m[(0,0)], 1);
assert_eq!(m[(1,1)], 5);
assert_eq!(m[(2,1)], 8);
``` */
pub trait Index2D: Copy + Debug {
    /** Convert an index to its 1-D linear interpretation, given the `width` and `height` of the
    matrix being subscripted.

    If the index is out of bounds for the given dimensions, this returns `None`,
    otherwise it returns `Some(usize)`

    # Examples
    ```
    # use vector_victor::index::Index2D;
    assert_eq!(
        (1usize,2usize).to_1d(3,3),
        Some(5),
        "(1,2) is index 5 in a 3×3 matrix");
    assert_eq!(
        (3usize, 2usize).to_1d(3,3),
        None,
        "row 3, column 2 is out of bounds for a 3×3 matrix");
    ``` */
    #[inline(always)]
    fn to_1d(self, height: usize, width: usize) -> Option<usize> {
        let (r, c) = self.to_2d(height, width)?;
        Some(r * width + c)
    }

    /** Convert an index to its 2-D interpretation, given the `width` and `height` of the
    matrix being subscripted.

    If the index is out of bounds for the given dimensions, this returns `None`,
    otherwise it returns `Some((usize, usize))`

    # Examples
    ```
    # use vector_victor::index::Index2D;
    assert_eq!(
        5usize.to_2d(3,3),
        Some((1usize,2usize)),
        "index 5 is at row 1 column 2 in a 3×3 matrix");
    assert_eq!(
        10usize.to_2d(3,3),
        None,
        "a 3×3 matrix only has 9 elements, so index 10 is out of bounds.");
    ``` */
    fn to_2d(self, height: usize, width: usize) -> Option<(usize, usize)>;
}

impl Index2D for usize {
    #[inline(always)]
    fn to_2d(self, height: usize, width: usize) -> Option<(usize, usize)> {
        match self < (height * width) {
            true => Some((self / width, self % width)),
            false => None,
        }
    }
}

impl Index2D for (usize, usize) {
    #[inline(always)]
    fn to_2d(self, height: usize, width: usize) -> Option<(usize, usize)> {
        match self.0 < height && self.1 < width {
            true => Some(self),
            false => None,
        }
    }
}
