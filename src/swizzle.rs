/** Rearrange the rows of a matrix by the given row identifiers
# Arguments

* `mat`: the matrix to manipulate
* `i...`: variadic comma-seperated list of row selectors. The row selectors are each one of:
    * const expression representing the row index of the input to copy. eg: `0` or `2`
    * a letter representing the same. you can use `r,g,b,a` or `x,y,z,w` to represent index 0 through 3,
      or `u,v` to represent indices 0 through 1.
    * an expression in curly braces, representing a value to be copied to an entire row of
      the result, eg: `{1.0}` or `{5}`

  note that the number of selectors doesnt need to match the height of the input!

# Examples
```
# use vector_victor::{swizzle, Vector};
let myvec = Vector::vec([0, 1, 2, 3]);

// each element can be selected with:
// 0: r, x, u, or 0
// 1: g, y, v, or 1
// 2: b, z, or 2
// 3: a, w, or 3
// or a result row can be filled by a new value

assert_eq!(swizzle!(myvec, a, z, v, 0, {7}), Vector::vec([3, 2, 1, 0, 7]));
```

More often you wont mix and match selector "flavors".
This example unpacks a [DXT5nm](http://wiki.polycount.com/wiki/Normal_Map_Compression) color
into the red and green channels, with blue filled with 0.
```
# use vector_victor::{swizzle, Vector};
let myvec = Vector::vec([0, 120, 0, 255]);
assert_eq!(swizzle!(myvec, a, g, {0}), Vector::vec([255, 120, 0]));
``` */
#[macro_export]
macro_rules! swizzle {
    ($mat: expr, $( $i:tt),+) => {{
        let mut result = $mat.permute_rows(&$crate::Vector::vec([$( $crate::get!($mat, $i), )+]));
        let mut _p = 0usize;
        $(
            $crate::sub_literal!(result, _p, $i);
            _p += 1;
        )+
        result
    }};
}

#[doc(hidden)]
#[macro_export]
macro_rules! get {
    ($mat:expr, x) => {
        0usize
    };
    ($mat:expr, y) => {
        1usize
    };
    ($mat:expr, z) => {
        2usize
    };
    ($mat:expr, w) => {
        3usize
    };
    ($mat:expr, r) => {
        0usize
    };
    ($mat:expr, g) => {
        1usize
    };
    ($mat:expr, b) => {
        2usize
    };
    ($mat:expr, a) => {
        3usize
    };
    ($mat:expr, u) => {
        0usize
    };
    ($mat:expr, v) => {
        1usize
    };
    ($mat:expr, {$i:expr}) => {
        0usize /* will be replaced later */
    };
    ($mat:expr, $i:expr) => {
        $i
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! sub_literal {
    ($result:expr, $p:expr, {$i:expr}) => {
        $result.set_row($p, &$crate::Vector::fill($i))
    };
    ($result:expr, $p:expr, $i:expr) => {};
}
