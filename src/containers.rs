use index::Index2D;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};

pub mod index;
pub mod iter;

/// This trait exists to allow structs like Slice2D to require Get2D, without
/// storing the dimensions of the target as part of its own generic parameters
pub trait Get2D {
    type Scalar: Sized + Copy;
    const HEIGHT: usize;
    const WIDTH: usize;

    fn get<I: Index2D>(&self, i: I) -> Option<&Self::Scalar>;
}

pub trait Get2DMut: Get2D {
    fn get_mut<I: Index2D>(&mut self, i: I) -> Option<&mut Self::Scalar>;
}

pub trait Get2DSized<const M: usize, const N: usize>: Get2D {}

/// A 2D owning array of T
#[derive(Debug, Clone, Copy)]
pub struct Array2D<T, const M: usize, const N: usize>
where
    T: Copy + 'static,
{
    pub data: [[T; N]; M],
}

impl<T, const M: usize, const N: usize> Default for Array2D<T, M, N>
where
    T: Default + Copy,
{
    fn default() -> Self {
        Array2D {
            data: [[T::default(); N]; M],
        }
    }
}

impl<T: Copy, const M: usize, const N: usize> Get2D for Array2D<T, M, N> {
    type Scalar = T;
    const HEIGHT: usize = M;
    const WIDTH: usize = N;

    fn get<I: Index2D>(&self, i: I) -> Option<&Self::Scalar> {
        let (r, c) = i.to_2d(Self::WIDTH);
        self.data.get(r)?.get(c)
    }
}

impl<T: Copy, const M: usize, const N: usize> Get2DMut for Array2D<T, M, N> {
    fn get_mut<I: Index2D>(&mut self, i: I) -> Option<&mut Self::Scalar> {
        let (r, c) = i.to_2d(Self::WIDTH);
        self.data.get_mut(r)?.get_mut(c)
    }
}

impl<T: Copy, const M: usize, const N: usize> Get2DSized<M, N> for Array2D<T, M, N> {}

/// A 2D mutable view into a container
#[derive(Debug, Clone, Copy)]
pub struct Slice2D<'a, R, const M: usize, const N: usize>
where
    R: Deref,
    R::Target: Get2D + 'a,
{
    target: R,
    r: usize,
    c: usize,
    phantom: PhantomData<&'a ()>,
}

impl<'a, R, const M: usize, const N: usize> Slice2D<'a, R, M, N>
where
    R: Deref,
    R::Target: Get2D + 'a,
{
    pub fn new(target: R, r: usize, c: usize) -> Self {
        Self {
            target,
            r,
            c,
            phantom: PhantomData,
        }
    }
}

impl<'a, R, D, const M: usize, const N: usize> Get2D for Slice2D<'a, R, M, N>
where
    R: Deref<Target = D>,
    D: Get2D,
{
    type Scalar = <<R as Deref>::Target as Get2D>::Scalar;
    const HEIGHT: usize = M;
    const WIDTH: usize = N;

    fn get<I: Index2D>(&self, i: I) -> Option<&Self::Scalar> {
        self.target
            .get(i.to_2d_offset(Self::WIDTH, Self::HEIGHT, self.r, self.c)?)
    }
}

impl<'a, R, D, const M: usize, const N: usize> Get2DMut for Slice2D<'a, R, M, N>
where
    R: Deref<Target = D> + DerefMut,
    D: Get2DMut,
{
    fn get_mut<I: Index2D>(&mut self, i: I) -> Option<&mut Self::Scalar> {
        self.target
            .get_mut(i.to_2d_offset(Self::WIDTH, Self::HEIGHT, self.r, self.c)?)
    }
}

impl<'a, R, D, const M: usize, const N: usize> Get2DSized<M, N> for Slice2D<'a, R, M, N>
where
    R: Deref<Target = D>,
    D: Get2D,
{
}

// A transposition of a 2D container
#[derive(Debug, Clone, Copy)]
pub struct Transpose<'a, R>
where
    R: Deref,
    R::Target: Get2D + 'a,
{
    target: R,
    phantom: std::marker::PhantomData<&'a ()>,
}

impl<'a, R> Transpose<'a, R>
where
    R: Deref,
    R::Target: Get2D + 'a,
{
    fn new(target: R) -> Self {
        Self {
            target,
            phantom: PhantomData,
        }
    }
}

impl<'a, R, D> Get2D for Transpose<'a, R>
where
    R: Deref<Target = D>,
    D: Get2D,
{
    type Scalar = D::Scalar;
    const HEIGHT: usize = D::WIDTH;
    const WIDTH: usize = D::HEIGHT;

    fn get<I: Index2D>(&self, i: I) -> Option<&Self::Scalar> {
        let (r, c) = i.to_2d(Self::WIDTH);
        self.target.get((c, r))
    }
}

impl<'a, R, D> Get2DMut for Transpose<'a, R>
where
    R: DerefMut<Target = D>,
    D: Get2DMut,
{
    fn get_mut<I: Index2D>(&mut self, i: I) -> Option<&mut Self::Scalar> {
        let (r, c) = i.to_2d(Self::WIDTH);
        self.target.get_mut((c, r))
    }
}

impl<'a, R, D, const M: usize, const N: usize> Get2DSized<M, N> for Transpose<'a, R>
where
    R: Deref<Target = D>,
    D: Get2DSized<N, M>,
{
}
