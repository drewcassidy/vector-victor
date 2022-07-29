use super::Get2D;
use std::iter::FusedIterator;

#[derive(Debug, Copy, Clone)]
pub(crate) struct ElementIter<'a, D>
where
    D: Get2D,
{
    data: &'a D,
    index: usize,
}

impl<'a, D: Get2D> ElementIter<'a, D> {
    pub(crate) fn new(data: &'a D) -> ElementIter<D> {
        ElementIter { data, index: 0 }
    }
}

impl<'a, D: Get2D> Iterator for ElementIter<'a, D> {
    type Item = &'a D::Scalar;

    fn next(&mut self) -> Option<Self::Item> {
        let result = self.data.get(self.index);
        self.index += 1;
        result
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let total = D::WIDTH * D::HEIGHT;
        (total - self.index, Some(total - self.index))
    }
}

impl<'a, D: Get2D> ExactSizeIterator for ElementIter<'a, D> {
    fn len(&self) -> usize {
        self.index - D::WIDTH * D::HEIGHT
    }
}
impl<'a, D: Get2D> FusedIterator for ElementIter<'a, D> {}
