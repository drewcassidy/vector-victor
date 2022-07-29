pub trait Index2D: Copy {
    fn to_1d(&self, width: usize) -> usize {
        let (r, c) = self.to_2d(width);
        r * width + c
    }

    fn to_2d(&self, width: usize) -> (usize, usize);

    fn to_2d_offset(
        &self,
        width: usize,
        height: usize,
        r: usize,
        c: usize,
    ) -> Option<(usize, usize)> {
        let (row, col) = self.to_2d(width);
        if row >= height || col >= width {
            return None;
        };
        Some((row + r, col + c))
    }
}

impl Index2D for usize {
    fn to_2d(&self, width: usize) -> (usize, usize) {
        (*self / width, *self % width)
    }
}

impl Index2D for (usize, usize) {
    fn to_2d(&self, _: usize) -> (usize, usize) {
        *self
    }
}
