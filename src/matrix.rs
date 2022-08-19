use crate::impl_matrix_op;
use crate::index::Index2D;
use std::fmt::Debug;
use std::iter::{zip, Flatten, Product, Sum};
use std::ops::{AddAssign, Deref, DerefMut, Index, IndexMut, MulAssign};

/// A Scalar that a [Matrix] can be made up of.
///
/// This trait has no associated functions and can be implemented on any type that is [Default] and
/// [Copy] and has a static lifetime.
pub trait Scalar: Default + Copy + 'static {}
macro_rules! multi_impl { ($name:ident for $($t:ty),*) => ($( impl $name for $t {} )*) }
multi_impl!(Scalar for i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize, f32, f64);
impl<T> Scalar for &'static T
where
    T: Scalar,
    &'static T: Default,
{
}

/// A 2D array of values which can be operated upon.
///
/// Matrices have a fixed size known at compile time, and must be made up of types that implement
/// the [Scalar] trait.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Matrix<T, const M: usize, const N: usize>
where
    T: Scalar,
{
    data: [[T; N]; M],
}

/// An alias for a [Matrix] with a single column
pub type Vector<T, const N: usize> = Matrix<T, N, 1>;

impl<T: Scalar, const M: usize, const N: usize> Matrix<T, M, N> {
    /// Generate a new matrix from a 2D Array
    ///
    /// # Arguments
    ///
    /// * `data`: A 2D array of elements to copy into the new matrix
    ///
    /// returns: Matrix<T, M, N>
    ///
    /// # Examples
    ///
    /// ```
    /// # use vector_victor::Matrix;
    /// let a = Matrix::new([[1,2,3,4];4]);
    /// ```
    #[must_use]
    pub fn new(data: [[T; N]; M]) -> Self {
        Matrix::<T, M, N> { data }
    }

    /// Generate a new matrix from a single scalar
    ///
    /// # Arguments
    ///
    /// * `scalar`: Scalar value to copy into the new matrix.
    ///
    /// returns: Matrix<T, M, N>
    ///
    /// # Examples
    ///
    /// ```
    /// # use vector_victor::Matrix;
    /// let my_matrix = Matrix::<i32,4,4>::fill(5);
    /// // is equivalent to
    /// assert_eq!(my_matrix, Matrix::new([[5;4];4]))
    /// ```
    #[must_use]
    pub fn fill(scalar: T) -> Matrix<T, M, N> {
        Matrix::<T, M, N> {
            data: [[scalar; N]; M],
        }
    }

    /// Create a matrix from an iterator of vectors
    ///
    /// # Arguments
    ///
    /// * `iter`: iterator of vectors to copy into rows
    ///
    /// returns: Matrix<T, M, N>
    ///
    /// # Examples
    ///
    /// ```
    /// # use vector_victor::Matrix;
    /// let my_matrix = Matrix::new([[1,2,3],[4,5,6]]);
    /// let transpose : Matrix<_,3,2>= Matrix::from_rows(my_matrix.cols());
    /// assert_eq!(transpose, Matrix::new([[1,4],[2,5],[3,6]]))
    /// ```
    #[must_use]
    pub fn from_rows<I>(iter: I) -> Self
    where
        Self: Default,
        I: IntoIterator<Item = Vector<T, N>>,
    {
        let mut result = Self::default();
        for (m, row) in iter.into_iter().enumerate().take(M) {
            result.set_row(m, &row)
        }
        result
    }

    /// Create a matrix from an iterator of vectors
    ///
    /// # Arguments
    ///
    /// * `iter`: iterator of vectors to copy into columns
    ///
    /// returns: Matrix<T, M, N>
    ///
    /// # Examples
    ///
    /// ```
    /// # use vector_victor::Matrix;
    /// let my_matrix = Matrix::new([[1,2,3],[4,5,6]]);
    /// let transpose : Matrix<_,3,2>= Matrix::from_cols(my_matrix.rows());
    /// assert_eq!(transpose, Matrix::new([[1,4],[2,5],[3,6]]))
    /// ```
    #[must_use]
    pub fn from_cols<I>(iter: I) -> Self
    where
        Self: Default,
        I: IntoIterator<Item = Vector<T, M>>,
    {
        let mut result = Self::default();
        for (n, col) in iter.into_iter().enumerate().take(N) {
            result.set_col(n, &col)
        }
        result
    }

    /// Returns an iterator over the elements of the matrix in row-major order.
    ///
    /// # Examples
    /// ```
    /// # use vector_victor::Matrix;
    /// let my_matrix = Matrix::new([[1,2],[3,4]]);
    /// assert!(vec![1,2,3,4].iter().eq(my_matrix.elements()))
    /// ```
    #[must_use]
    pub fn elements<'a>(&'a self) -> impl Iterator<Item = &T> + 'a {
        self.data.iter().flatten()
    }

    /// Returns a mutable iterator over the elements of the matrix in row-major order.
    #[must_use]
    pub fn elements_mut<'a>(&'a mut self) -> impl Iterator<Item = &mut T> + 'a {
        self.data.iter_mut().flatten()
    }

    /// Returns a reference to the element at that position in the matrix or `None` if out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// # use vector_victor::Matrix;
    /// let my_matrix = Matrix::new([[1,2],[3,4]]);
    ///
    /// // element at index 2 is the same as the element at (row 1, column 0).
    /// assert_eq!(my_matrix.get(2), my_matrix.get((1,0)));
    /// // index 4 is out of range, so get(4) returns None.
    /// assert_eq!(my_matrix.get(4), None);
    /// ```
    #[inline]
    #[must_use]
    pub fn get(&self, index: impl Index2D) -> Option<&T> {
        let (m, n) = index.to_2d(M, N)?;
        Some(&self.data[m][n])
    }

    #[inline]
    #[must_use]
    pub fn get_mut(&mut self, index: impl Index2D) -> Option<&mut T> {
        let (m, n) = index.to_2d(M, N)?;
        Some(&mut self.data[m][n])
    }

    #[inline]
    #[must_use]
    pub fn row(&self, m: usize) -> Option<Vector<T, N>> {
        if m < M {
            Some(Vector::<T, N>::vec(self.data[m]))
        } else {
            None
        }
    }

    #[inline]
    pub fn set_row(&mut self, m: usize, val: &Vector<T, N>) {
        assert!(
            m < M,
            "Row index {} out of bounds for {}x{} matrix",
            m,
            M,
            N
        );
        for (n, v) in val.elements().enumerate() {
            self.data[m][n] = *v;
        }
    }

    #[inline]
    #[must_use]
    pub fn col(&self, n: usize) -> Option<Vector<T, M>> {
        if n < N {
            Some(Vector::<T, M>::vec(self.data.map(|r| r[n])))
        } else {
            None
        }
    }

    #[inline]
    pub fn set_col(&mut self, n: usize, val: &Vector<T, M>) {
        assert!(
            n < N,
            "Column index {} out of bounds for {}x{} matrix",
            n,
            M,
            N
        );

        for (m, v) in val.elements().enumerate() {
            self.data[m][n] = *v;
        }
    }

    #[must_use]
    pub fn rows<'a>(&'a self) -> impl Iterator<Item = Vector<T, N>> + 'a {
        (0..M).map(|m| self.row(m).expect("invalid row reached while iterating"))
    }

    #[must_use]
    pub fn cols<'a>(&'a self) -> impl Iterator<Item = Vector<T, M>> + 'a {
        (0..N).map(|n| self.col(n).expect("invalid column reached while iterating"))
    }
}

// 1D vector implementations
impl<T: Scalar, const M: usize> Matrix<T, M, 1> {
    /// Create a vector from a 1D array.
    /// Note that vectors are always column vectors unless explicitly instantiated as row vectors
    ///
    /// # Arguments
    ///
    /// * `data`: A 1D array of elements to copy into the new vector
    ///
    /// returns: Matrix<T, { M }, 1>
    ///
    /// # Examples
    ///
    /// ```
    /// # use vector_victor::{Matrix, Vector};
    /// let my_vector = Vector::vec([1,2,3,4]);
    /// // is equivalent to
    /// assert_eq!(my_vector, Matrix::new([[1],[2],[3],[4]]));
    /// ```
    pub fn vec(data: [T; M]) -> Self {
        return Matrix::<T, M, 1> {
            data: data.map(|e| [e; 1]),
        };
    }
}

// Index
impl<I, T, const M: usize, const N: usize> Index<I> for Matrix<T, M, N>
where
    I: Index2D,
    T: Scalar,
{
    type Output = T;

    fn index(&self, index: I) -> &Self::Output {
        self.get(index).expect(&*format!(
            "index {:?} out of range for {}x{} Matrix",
            index, M, N
        ))
    }
}

// IndexMut
impl<I, T, const M: usize, const N: usize> IndexMut<I> for Matrix<T, M, N>
where
    I: Index2D,
    T: Scalar,
{
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        self.get_mut(index).expect(&*format!(
            "index {:?} out of range for {}x{} Matrix",
            index, M, N
        ))
    }
}

// Default
impl<T: Scalar, const M: usize, const N: usize> Default for Matrix<T, M, N> {
    fn default() -> Self {
        Matrix {
            data: [[T::default(); N]; M],
        }
    }
}

impl<T: Scalar, const M: usize, const N: usize> From<[[T; N]; M]> for Matrix<T, M, N> {
    fn from(data: [[T; N]; M]) -> Self {
        Self::new(data)
    }
}

impl<T: Scalar, const M: usize> From<[T; M]> for Vector<T, M> {
    fn from(data: [T; M]) -> Self {
        Self::vec(data)
    }
}

impl<T: Scalar, const M: usize, const N: usize> From<T> for Matrix<T, M, N> {
    fn from(scalar: T) -> Self {
        Self::fill(scalar)
    }
}

// deref 1x1 matrices to a scalar automatically
impl<T: Scalar> Deref for Matrix<T, 1, 1> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.data[0][0]
    }
}

// deref 1x1 matrices to a mutable scalar automatically
impl<T: Scalar> DerefMut for Matrix<T, 1, 1> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data[0][0]
    }
}

// IntoIter
impl<T: Scalar, const M: usize, const N: usize> IntoIterator for Matrix<T, M, N> {
    type Item = T;
    type IntoIter = Flatten<std::array::IntoIter<[T; N], M>>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter().flatten()
    }
}

// FromIterator
impl<T: Scalar, const M: usize, const N: usize> FromIterator<T> for Matrix<T, M, N>
where
    Self: Default,
{
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut result: Self = Default::default();
        for (l, r) in zip(result.elements_mut(), iter) {
            *l = r;
        }
        result
    }
}

impl<T: Scalar + AddAssign, const M: usize, const N: usize> Sum for Matrix<T, M, N> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut sum = Self::default();

        for m in iter {
            sum += m;
        }

        sum
    }
}

impl<T: Scalar + MulAssign, const M: usize, const N: usize> Product for Matrix<T, M, N> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut prod = Self::default();

        for m in iter {
            prod *= m;
        }

        prod
    }
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
