use crate::{Matrix, Vector};
use num_traits::real::Real;
use num_traits::{Inv, Num, NumOps, One, Pow, Signed, Zero};
use std::iter::{zip, Product, Sum};
use std::ops::{Add, Mul};

/// Operations for column vectors
impl<T: Copy, const N: usize> Vector<T, N> {
    /** Compute the dot product of two vectors, otherwise known as the scalar product.

    This is the sum of the elementwise product, or in math terms

    $vec(a) * vec(b) = sum_(i=1)^n a_i b_i = a_1 b_1 + a_2 b_2 + ... + a_n b_n$

    for example, $\[\[1],\[2],\[3]] * \[\[4],\[5],\[6]] = (1 * 4) + (2 * 5) + (3 * 6) = 32$

    For vectors in euclidean space, this has the property that it is equal to the magnitudes of
    the vectors times the cosine of the angle between them.

    $vec(a) * vec(b) = |vec(a)| |vec(b)| cos(theta)$

    this also gives it the special property that the dot product of a vector and itself is the
    square of its magnitude. You may recognize the 2D version as the
    [pythagorean theorem](https://en.wikipedia.org/wiki/Pythagorean_theorem).

    see [dot product](https://en.wikipedia.org/wiki/Dot_product) on Wikipedia for more
    information. */
    pub fn dot<R>(&self, rhs: &R) -> T
    where
        for<'s> &'s Self: Mul<&'s R, Output = Self>,
        T: Sum<T>,
    {
        (self * rhs).elements().cloned().sum()
    }

    pub fn sqrmag(&self) -> T
    where
        for<'s> &'s Self: Mul<&'s Self, Output = Self>,
        T: Sum<T>,
    {
        self.dot(self)
    }

    pub fn mag(&self) -> T
    where
        T: Sum<T> + Mul<T> + Real,
    {
        self.sqrmag().sqrt()
    }

    pub fn normalized(&self) -> Option<Self>
    where
        T: Sum<T> + Mul<T> + Real,
    {
        match self.mag() {
            mag if mag.abs() < T::epsilon() => None,
            mag => Some(self / mag),
        }
    }
}

/// Cross product operations for column vectors in $RR^3$
impl<T: Copy> Vector<T, 3> {
    pub fn cross_r<R: Copy>(&self, rhs: &Vector<R, 3>) -> Self
    where
        T: NumOps<R> + NumOps,
    {
        Self::vec([
            (self[1] * rhs[2]) - (self[2] * rhs[1]),
            (self[2] * rhs[0]) - (self[0] * rhs[2]),
            (self[0] * rhs[1]) - (self[1] * rhs[0]),
        ])
    }

    pub fn cross_l<R: Copy>(&self, rhs: &Vector<R, 3>) -> Vector<R, 3>
    where
        R: NumOps<T> + NumOps,
    {
        rhs.cross_r(self)
    }
}

/// Operations for Matrices
impl<T: Copy, const M: usize, const N: usize> Matrix<T, M, N> {
    pub fn mmul<R: Copy, const P: usize>(&self, rhs: &Matrix<R, N, P>) -> Matrix<T, M, P>
    where
        T: Default + NumOps<R> + Sum,
    {
        let mut result: Matrix<T, M, P> = Default::default();

        for (m, a) in self.rows().enumerate() {
            for (n, b) in rhs.cols().enumerate() {
                result[(m, n)] = a.dot(&b)
            }
        }

        return result;
    }

    /// Computes the absolute value of each element of the matrix
    pub fn abs(&self) -> Self
    where
        T: Signed + Default,
    {
        self.elements().map(|&x| x.abs()).collect()
    }

    /// Computes the sign of each element of the matrix
    pub fn signum(&self) -> Self
    where
        T: Signed + Default,
    {
        self.elements().map(|&x| x.signum()).collect()
    }

    /// Raises every element to the power of rhs, where rhs is either a scalar or a matrix of exponents
    pub fn pow<R, O>(self, rhs: R) -> O
    where
        Self: Pow<R, Output = O>,
    {
        Pow::pow(self, rhs)
    }
}

// Sum up matrices
impl<T: Copy, const M: usize, const N: usize> Sum for Matrix<T, M, N>
where
    Self: Zero + Add<Output = Self>,
{
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), Self::add)
    }
}

// Product of matrices
impl<T: Copy, const M: usize, const N: usize> Product for Matrix<T, M, N>
where
    Self: One + Mul<Output = Self>,
{
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::one(), Self::mul)
    }
}

/// Inverse trait. Note that this is the elementwise inverse, not the matrix inverse.
/// For the inverse matrix see [`LUDecomposable::inv()`](crate::decompose::LUDecompose::inv())
impl<T: Copy + Inv<Output = T> + Default, const M: usize, const N: usize> Inv for Matrix<T, M, N> {
    type Output = Self;

    fn inv(self) -> Self::Output {
        self.elements().map(|t| t.inv()).collect()
    }
}

/// Pow for $Matrix^{scalar}$
impl<T, R, O, const M: usize, const N: usize> Pow<R> for Matrix<T, M, N>
where
    T: Copy + Pow<R, Output = O>,
    R: Copy + Num,
    O: Copy + Default,
{
    type Output = Matrix<O, M, N>;

    fn pow(self, rhs: R) -> Self::Output {
        self.elements().map(|&x| x.pow(rhs)).collect()
    }
}

/// Pow for $Matrix^{Matrix}$
impl<T, R, O, const M: usize, const N: usize> Pow<Matrix<R, M, N>> for Matrix<T, M, N>
where
    T: Copy + Pow<R, Output = O>,
    R: Copy,
    O: Copy + Default,
{
    type Output = Matrix<O, M, N>;

    fn pow(self, rhs: Matrix<R, M, N>) -> Self::Output {
        zip(self.elements(), rhs.elements())
            .map(|(x, &r)| x.pow(r))
            .collect()
    }
}
