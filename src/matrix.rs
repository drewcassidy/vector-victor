use crate::containers::index::Index2D;
use crate::containers::iter::ElementIter;
use crate::containers::{Array2D, Get2D, Get2DMut, Get2DSized, Slice2D};
use std::ops::{Add, Index, IndexMut};

type Matrix<T: Copy, const M: usize, const N: usize> = GenericMatrix<Array2D<T, M, N>, M, N>;

#[derive(Debug, Copy, Clone)]
struct GenericMatrix<D: Get2DSized<M, N>, const M: usize, const N: usize> {
    data: D,
}

impl<D: Get2DSized<M, N>, const M: usize, const N: usize> GenericMatrix<D, M, N> {
    fn elements(&self) -> ElementIter<GenericMatrix<D, M, N>> {
        ElementIter::new(self)
    }
}

impl<D: Get2DSized<M, N> + Copy, const M: usize, const N: usize> Default for GenericMatrix<D, M, N>
where
    D: Default,
{
    fn default() -> Self {
        GenericMatrix { data: D::default() }
    }
}
// impl<D: Get2D + Copy, const M: usize, const N: usize> Matrix<D, M, N>
// where
//     D::Scalar: Default,
// {
//     fn new(data: &[&[D::Scalar]]) -> Result<Self, &'static str> {}
// }

impl<D: Get2DSized<M, N>, const M: usize, const N: usize> Get2D for GenericMatrix<D, M, N> {
    type Scalar = D::Scalar;
    const HEIGHT: usize = D::HEIGHT;
    const WIDTH: usize = D::WIDTH;

    fn get<I: Index2D>(&self, i: I) -> Option<&Self::Scalar> {
        self.data.get(i)
    }
}

impl<D: Get2DMut + Get2DSized<M, N>, const M: usize, const N: usize> Get2DMut
    for GenericMatrix<D, M, N>
{
    fn get_mut<I: Index2D>(&mut self, i: I) -> Option<&mut Self::Scalar> {
        self.data.get_mut(i)
    }
}

impl<D: Get2DSized<M, N>, I: Index2D, const M: usize, const N: usize> Index<I>
    for GenericMatrix<D, M, N>
{
    type Output = D::Scalar;

    fn index(&self, index: I) -> &Self::Output {
        self.get(index).expect(&*format!(
            "Index {:?} out of range for {} x {} matrix",
            index.to_2d(D::WIDTH),
            D::HEIGHT,
            D::WIDTH
        ))
    }
}

impl<D: Get2DMut + Get2DSized<M, N>, I: Index2D, const M: usize, const N: usize> IndexMut<I>
    for GenericMatrix<D, M, N>
{
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        self.get_mut(index).expect(&*format!(
            "Index {:?} out of range for {} x {} matrix",
            index.to_2d(D::WIDTH),
            D::HEIGHT,
            D::WIDTH
        ))
    }
}

impl<D: Get2DSized<M, N>, const M: usize, const N: usize> Get2DSized<M, N>
    for GenericMatrix<D, M, N>
{
}

fn foo() {
    let mut a: Matrix<i32, 5, 5> = Default::default();
    let c = Slice2D::<&Matrix<i32, 5, 5>, 3, 3>::new(&a, 2, 2);
    let b = Slice2D::<&mut Matrix<i32, 5, 5>, 3, 3>::new(&mut a, 1, 1);
    println!("{:?} {:?}", b, c)
}
