use crate::index2d::Index2D;

pub trait Container2D {
    type Output;
    const HEIGHT: u32;
    const WIDTH: u32;

    fn get<I: Index2D>(&self, i: I) -> Option<&Self::Output>;
}

pub trait Container2DMut: Container2D {
    fn get_mut<I: Index2D>(&mut self, i: I) -> Option<&mut Self::Output>;
}

/// A 2D owning array of T
#[derive(Debug)]
pub struct Array2D<T, const M: u32, const N: u32> {
    pub data: [[T; N as usize]; M as usize],
}

impl<T, const M: u32, const N: u32> Container2D for Array2D<T, M, N> {
    type Output = T;
    const HEIGHT: u32 = M;
    const WIDTH: u32 = N;

    fn get<I: Index2D>(&self, i: I) -> Option<&Self::Output> {
        let (r, c) = i.to_2d(Self::WIDTH);
        self.data.get(r)?.get(c)
    }
}

impl<T, const M: u32, const N: u32> Container2DMut for Array2D<T, M, N> {
    fn get_mut<I: Index2D>(&mut self, i: I) -> Option<&mut Self::Output> {
        let (r, c) = i.to_2d(Self::WIDTH);
        self.data.get_mut(r)?.get_mut(c)
    }
}

/// A 2D immutable view into a Container2D
#[derive(Debug)]
pub struct View2D<'a, D: Container2D, const M: u32, const N: u32> {
    r: u32,
    c: u32,
    data: &'a D,
}

impl<'a, D: Container2D, const M: u32, const N: u32> Container2D for View2D<'a, D, M, N> {
    type Output = D::Output;
    const HEIGHT: u32 = M;
    const WIDTH: u32 = N;

    fn get<I: Index2D>(&self, i: I) -> Option<&Self::Output> {
        self.data.get(i.to_2d_offset(Self::WIDTH, Self::HEIGHT, self.r, self.c)?)
    }
}

impl<'a, D: Container2DMut, const M: u32, const N: u32> Container2D for Slice2D<'a, D, M, N> {
    type Output = D::Output;
    const HEIGHT: u32 = M;
    const WIDTH: u32 = N;

    fn get<I: Index2D>(&self, i: I) -> Option<&Self::Output> {
        self.data.get(i.to_2d_offset(Self::WIDTH, Self::HEIGHT, self.r, self.c)?)
    }
}

/// A 2D mutable view into a Container2D
#[derive(Debug)]
pub struct Slice2D<'a, D: Container2DMut, const M: u32, const N: u32> {
    r: u32,
    c: u32,
    data: &'a mut D,
}

impl<'a, D: Container2DMut, const M: u32, const N: u32> Container2DMut
for Slice2D<'a, D, M, N>
{
    fn get_mut<I: Index2D>(&mut self, i: I) -> Option<&mut Self::Output> {
        self.data.get_mut(i.to_2d_offset(Self::WIDTH, Self::HEIGHT, self.r, self.c)?)
    }
}

// An immutable transposition of a Container2D
#[derive(Debug)]
pub struct Transpose<'a, D: Container2D> {
    pub data: &'a D,
}

impl<'a, D: Container2D> Container2D for Transpose<'a, D> {
    type Output = D::Output;
    const HEIGHT: u32 = D::WIDTH;
    const WIDTH: u32 = D::HEIGHT;

    fn get<I: Index2D>(&self, i: I) -> Option<&Self::Output> {
        let (r, c) = i.to_2d(Self::WIDTH);
        self.data.get((c, r))
    }
}
