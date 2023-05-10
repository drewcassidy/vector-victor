use crate::util::checked_inv;
use crate::{Matrix, Vector};
use num_traits::real::Real;
use std::iter::{Product, Sum};
use std::ops::{Mul, Neg, Not};

/// The parity of an [LU decomposition](LUDecomposition). In other words, how many times the
/// source matrix has to have rows swapped before the decomposition takes place
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Parity {
    Even,
    Odd,
}

impl<T> Mul<T> for Parity
where
    T: Neg<Output = T>,
{
    type Output = T;

    fn mul(self, rhs: T) -> Self::Output {
        match self {
            Parity::Even => rhs,
            Parity::Odd => -rhs,
        }
    }
}

impl Not for Parity {
    type Output = Parity;

    fn not(self) -> Self::Output {
        match self {
            Parity::Even => Parity::Odd,
            Parity::Odd => Parity::Even,
        }
    }
}

/// The result of the [LU decomposition](LUDecomposable::lu) of a matrix.
///
/// This struct provides a convenient way to reuse one LU decomposition to solve multiple
/// matrix equations. You likely do not need to worry about its contents.
///
/// See [LU decomposition](https://en.wikipedia.org/wiki/LU_decomposition)
/// on wikipedia for more information
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct LUDecomposition<T: Copy, const N: usize> {
    /// The $L$ and $U$ matrices combined into one
    ///
    /// for example if
    ///
    /// $ U = [[u_{11}, u_{12}, cdots,  u_{1n} ],
    ///        [0,      u_{22}, cdots,  u_{2n} ],
    ///        [vdots,  vdots,  ddots,  vdots  ],
    ///        [0,      0,      cdots,  u_{mn} ]] $
    /// and
    /// $ L = [[1,      0,      cdots,  0      ],
    ///        [l_{21}, 1,      cdots,  0      ],
    ///        [vdots,  vdots,  ddots,  vdots  ],
    ///        [l_{m1}, l_{m2}, cdots,  1      ]] $,
    /// then
    /// $ LU = [[u_{11}, u_{12}, cdots,  u_{1n} ],
    ///         [l_{21}, u_{22}, cdots,  u_{2n} ],
    ///         [vdots,  vdots,  ddots,  vdots  ],
    ///         [l_{m1}, l_{m2}, cdots,  u_{mn} ]] $
    ///
    /// note that the diagonals of the $L$ matrix are always 1, so no information is lost
    pub lu: Matrix<T, N, N>,

    /// The indices of the permutation matrix $P$, such that $PxxA$ = $LxxU$
    ///
    /// The permutation matrix rearranges the rows of the original matrix in order to produce
    /// the LU decomposition. This makes calculation simpler, but makes the result
    /// (known as an LUP decomposition) no longer unique
    pub idx: Vector<usize, N>,

    /// The parity of the decomposition.
    pub parity: Parity,
}

impl<T: Copy + Default, const N: usize> LUDecomposition<T, N>
where
    T: Real + Default + Sum + Product,
{
    /// Solve for $x$ in $M xx x = b$, where $M$ is the original matrix this is a decomposition of.
    ///
    /// This is equivalent to [`LUDecomposable::solve`] while allowing the LU decomposition
    /// to be reused
    #[must_use]
    pub fn solve<const M: usize>(&self, b: &Matrix<T, N, M>) -> Matrix<T, N, M> {
        let b_permuted = b.permute_rows(&self.idx);

        Matrix::from_cols(b_permuted.cols().map(|mut x| {
            // Implementation from Numerical Recipes §2.3
            // When ii is set to a positive value,
            // it will become the index of the first non-vanishing element of b
            let mut ii = 0usize;
            for i in 0..N {
                // forward substitution using L
                let mut sum = x[i];
                if ii != 0 {
                    for j in (ii - 1)..i {
                        sum = sum - (self.lu[(i, j)] * x[j]);
                    }
                } else if sum.abs() > T::epsilon() {
                    ii = i + 1;
                }
                x[i] = sum;
            }
            for i in (0..N).rev() {
                // back substitution using U
                let mut sum = x[i];
                for j in (i + 1)..N {
                    sum = sum - (self.lu[(i, j)] * x[j]);
                }
                x[i] = sum / self.lu[(i, i)]
            }
            x
        }))
    }

    /// Calculate the determinant $|M|$ of the matrix $M$.
    /// If the matrix is singular, the determinant is 0.
    ///
    /// This is equivalent to [`LUDecomposable::det`] while allowing the LU decomposition
    /// to be reused
    pub fn det(&self) -> T {
        self.parity * self.lu.diagonals().product()
    }

    /// Calculate the inverse of the original matrix, such that $MxxM^{-1} = I$
    ///
    /// This is equivalent to [`Matrix::inverse`] while allowing the LU decomposition to be reused
    #[must_use]
    pub fn inverse(&self) -> Matrix<T, N, N> {
        return self.solve(&Matrix::<T, N, N>::identity());
    }

    /// Separate the $L$ and $U$ sides of the $LU$ matrix.
    /// See [the `lu` field](LUDecomposition::lu) for more information
    pub fn separate(&self) -> (Matrix<T, N, N>, Matrix<T, N, N>) {
        let mut l = Matrix::<T, N, N>::identity();
        let mut u = self.lu; // lu

        for m in 1..N {
            for n in 0..m {
                // iterate over lower diagonal
                l[(m, n)] = u[(m, n)];
                u[(m, n)] = T::zero();
            }
        }

        (l, u)
    }
}

/// A Matrix that can be decomposed into an upper and lower diagonal matrix,
/// known as an [LU Decomposition](LUDecomposition).
///
/// See [LU decomposition](https://en.wikipedia.org/wiki/LU_decomposition)
/// on wikipedia for more information
pub trait LUDecomposable<T, const N: usize>
where
    T: Copy + Default + Real + Product + Sum,
{
    /// return this matrix's [`LUDecomposition`], or [`None`] if the matrix is singular.
    /// This can be used to solve for multiple results
    ///
    /// ```
    /// # use vector_victor::decompose::LUDecomposable;
    /// # use vector_victor::{Matrix, Vector};
    /// let m = Matrix::new([[1.0,3.0],[2.0,4.0]]);
    /// let lu = m.lu().expect("Cannot decompose a signular matrix");
    ///
    /// let b = Vector::vec([7.0,10.0]);
    /// assert_eq!(lu.solve(&b), Vector::vec([1.0,2.0]));
    ///
    /// let c = Vector::vec([10.0, 14.0]);
    /// assert_eq!(lu.solve(&c), Vector::vec([1.0,3.0]));
    ///
    /// ```
    #[must_use]
    fn lu(&self) -> Option<LUDecomposition<T, N>>;

    /// Calculate the inverse of the matrix, such that $MxxM^{-1} = I$, or [`None`] if the matrix is singular.
    ///
    /// ```
    /// # use vector_victor::decompose::LUDecomposable;
    /// # use vector_victor::Matrix;
    /// let m = Matrix::new([[1.0,3.0],[2.0,4.0]]);
    /// let mi = m.inverse().expect("Cannot invert a singular matrix");
    ///
    /// assert_eq!(mi, Matrix::new([[-2.0, 1.5],[1.0, -0.5]]), "unexpected inverse matrix");
    ///
    /// // multiplying a matrix by its inverse yields the identity matrix
    /// assert_eq!(m.mmul(&mi), Matrix::identity())
    /// ```
    #[must_use]
    fn inverse(&self) -> Option<Matrix<T, N, N>>;

    /// Calculate the determinant $|M|$ of the matrix $M$.
    /// If the matrix is singular, the determinant is 0
    #[must_use]
    fn det(&self) -> T;

    /// Solve for $x$ in $M xx x = b$
    ///
    /// ```
    /// # use vector_victor::decompose::LUDecomposable;
    /// # use vector_victor::{Matrix, Vector};
    ///
    /// let m = Matrix::new([[1.0,3.0],[2.0,4.0]]);
    /// let b = Vector::vec([7.0,10.0]);
    /// let x = m.solve(&b).expect("Cannot solve a singular matrix");
    ///
    /// assert_eq!(x, Vector::vec([1.0,2.0]), "x = [1,2]");
    /// assert_eq!(m.mmul(&x), b, "Mx = b");
    /// ```
    ///
    /// $x$ does not need to be a column-vector, it can also be a 2D matrix. For example,
    /// the following is another way to calculate the [inverse](LUDecomposable::inverse()) by solving for the identity matrix $I$.
    ///
    /// ```
    /// # use vector_victor::decompose::LUDecomposable;
    /// # use vector_victor::{Matrix, Vector};
    ///
    /// let m = Matrix::new([[1.0,3.0],[2.0,4.0]]);
    /// let i = Matrix::<f64,2,2>::identity();
    /// let mi = m.solve(&i).expect("Cannot solve a singular matrix");
    ///
    /// assert_eq!(mi, Matrix::new([[-2.0, 1.5],[1.0, -0.5]]));
    /// assert_eq!(m.mmul(&mi), i, "M x M^-1 = I");
    /// ```
    #[must_use]
    fn solve<const M: usize>(&self, b: &Matrix<T, N, M>) -> Option<Matrix<T, N, M>>;
}

impl<T, const N: usize> LUDecomposable<T, N> for Matrix<T, N, N>
where
    T: Copy + Default + Real + Sum + Product,
{
    fn lu(&self) -> Option<LUDecomposition<T, N>> {
        // Implementation from Numerical Recipes §2.3
        let mut lu = self.clone();
        let mut idx: Vector<usize, N> = (0..N).collect();
        let mut parity = Parity::Even;

        let mut vv: Vector<T, N> = self
            .rows()
            .map(|row| {
                let m = row.elements().cloned().reduce(|acc, x| acc.max(x.abs()))?;
                checked_inv(m)
            })
            .collect::<Option<_>>()?; // get the inverse max abs value in each row

        // for each column in the matrix...
        for k in 0..N {
            // search for the pivot element and its index
            let (ipivot, _) = (lu.col(k) * vv)
                .abs()
                .elements()
                .enumerate()
                .skip(k) // below the diagonal
                .reduce(|(imax, xmax), (i, x)| match x > xmax {
                    // Is the figure of merit for the pivot better than the best so far?
                    true => (i, x),
                    false => (imax, xmax),
                })?;

            // do we need to interchange rows?
            if k != ipivot {
                lu.pivot_row(ipivot, k); // yes, we do
                idx.pivot_row(ipivot, k);
                parity = !parity; // swap parity
                vv[ipivot] = vv[k] // interchange scale factor
            }

            // select our pivot, which is now on the diagonal
            let pivot = lu[(k, k)];
            if pivot.abs() < T::epsilon() {
                // if the pivot is zero, the matrix is singular
                return None;
            };

            // for each element in the column k below the diagonal...
            // this is called outer product Gaussian elimination
            for i in (k + 1)..N {
                // divide by the pivot element
                lu[(i, k)] = lu[(i, k)] / pivot;

                // for each element in the column k below the diagonal...
                for j in (k + 1)..N {
                    // reduce remaining submatrix
                    lu[(i, j)] = lu[(i, j)] - (lu[(i, k)] * lu[(k, j)]);
                }
            }
        }

        return Some(LUDecomposition { lu, idx, parity });
    }

    fn inverse(&self) -> Option<Matrix<T, N, N>> {
        match N {
            1 => Some(Self::fill(checked_inv(self[0])?)),
            2 => {
                let mut result = Self::default();
                result[(0, 0)] = self[(1, 1)];
                result[(1, 1)] = self[(0, 0)];
                result[(1, 0)] = -self[(1, 0)];
                result[(0, 1)] = -self[(0, 1)];
                Some(result * checked_inv(self.det())?)
            }
            _ => Some(self.lu()?.inverse()),
        }
    }

    fn det(&self) -> T {
        match N {
            1 => self[0],
            2 => (self[(0, 0)] * self[(1, 1)]) - (self[(0, 1)] * self[(1, 0)]),
            3 => {
                // use rule of Sarrus
                (0..N) // starting column
                    .map(|i| {
                        let dn = (0..N)
                            .map(|j| -> T { self[(j, (j + i) % N)] })
                            .product::<T>();
                        let up = (0..N)
                            .map(|j| -> T { self[(N - j - 1, (j + i) % N)] })
                            .product::<T>();
                        dn - up
                    })
                    .sum::<T>()
            }
            _ => {
                // use LU decomposition
                self.lu().map_or(T::zero(), |lu| lu.det())
            }
        }
    }

    fn solve<const M: usize>(&self, b: &Matrix<T, N, M>) -> Option<Matrix<T, N, M>> {
        Some(self.lu()?.solve(b))
    }
}
