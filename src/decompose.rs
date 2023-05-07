use crate::util::checked_inv;
use crate::Matrix;
use crate::Vector;
use num_traits::real::Real;
use std::iter::{Product, Sum};

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct LUDecomposition<T: Copy, const N: usize> {
    pub lu: Matrix<T, N, N>,
    pub idx: Vector<usize, N>,
    pub parity: T,
}

impl<T: Copy + Default, const N: usize> LUDecomposition<T, N>
where
    T: Real + Default + Sum + Product,
{
    #[must_use]
    pub fn decompose(m: &Matrix<T, N, N>) -> Option<Self> {
        // Implementation from Numerical Recipes §2.3
        let mut lu = m.clone();
        let mut idx: Vector<usize, N> = (0..N).collect();
        let mut parity = T::one();

        let mut vv: Vector<T, N> = m
            .rows()
            .map(|row| {
                let m = row.elements().cloned().reduce(|acc, x| acc.max(x.abs()))?;
                match m < T::epsilon() {
                    true => None,
                    false => Some(T::one() / m),
                }
            })
            .collect::<Option<_>>()?; // get the inverse maxabs value in each row

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
                parity = -parity; // change parity of d
                vv[ipivot] = vv[k] //interchange scale factor
            }

            let pivot = lu[(k, k)];
            if pivot.abs() < T::epsilon() {
                // if the pivot is zero, the matrix is singular
                return None;
            };

            for i in (k + 1)..N {
                // divide by the pivot element
                let dpivot = lu[(i, k)] / pivot;
                lu[(i, k)] = dpivot;
                for j in (k + 1)..N {
                    // reduce remaining submatrix
                    lu[(i, j)] = lu[(i, j)] - (dpivot * lu[(k, j)]);
                }
            }
        }

        return Some(Self { lu, idx, parity });
    }

    #[must_use]
    pub fn solve<const M: usize>(&self, b: &Matrix<T, N, M>) -> Matrix<T, N, M> {
        let b_permuted = b.permute_rows(&self.idx);

        Matrix::from_cols(b_permuted.cols().map(|mut x| {
            // Implementation from Numerical Recipes §2.3
            // When ii is set to a positive value,
            // it will become the index of the first nonvanishing element of b
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

    pub fn det(&self) -> T {
        self.parity * self.lu.diagonals().product()
    }

    pub fn inverse(&self) -> Matrix<T, N, N> {
        return self.solve(&Matrix::<T, N, N>::identity());
    }

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

pub trait LUDecomposable<T, const N: usize>
where
    T: Copy + Default + Real + Product + Sum,
{
    #[must_use]
    fn lu(&self) -> Option<LUDecomposition<T, N>>;

    #[must_use]
    fn inverse(&self) -> Option<Matrix<T, N, N>>;

    #[must_use]
    fn det(&self) -> T;

    #[must_use]
    fn solve<const M: usize>(&self, b: &Matrix<T, N, M>) -> Option<Matrix<T, N, M>>;
}

impl<T, const N: usize> LUDecomposable<T, N> for Matrix<T, N, N>
where
    T: Copy + Default + Real + Sum + Product,
{
    fn lu(&self) -> Option<LUDecomposition<T, N>> {
        LUDecomposition::decompose(self)
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
