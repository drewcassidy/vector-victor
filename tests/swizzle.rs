// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use vector_victor::legacy::Matrix;
use vector_victor::legacy::Vector;
use vector_victor::swizzle;

#[test]
fn test_swizzle() {
    let identity = Matrix::<i32, 4, 4>::identity();

    assert_eq!(
        swizzle!(identity, a, x, b, (1 + 0), { 2 - 3 }),
        Matrix::mat([
            [0, 0, 0, 1],     // a
            [1, 0, 0, 0],     // x
            [0, 0, 1, 0],     // b
            [0, 1, 0, 0],     // row 1
            [-1, -1, -1, -1]  // fill(-1)
        ])
    );
}
