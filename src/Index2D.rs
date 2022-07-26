pub trait Index2D {
    fn to_1d(&self, width: u32) -> u32 {
        let (r, c) = self.to_2d(width);
        r * width + c
    }

    fn to_2d(&self, width: u32) -> (u32, u32);

    fn to_2d_offset(&self, width: u32, height: u32, r: u32, c: u32) -> Option<(u32, u32)> {
        let (row, col) = self.to_2d(width);
        if row >= height || col >= width {
            return None;
        };
        Some((row + r, col + c))
    }
}

impl Index2D for u32 {
    fn to_2d(&self, width: u32) -> (u32, u32) {
        (*self / width, *self % width)
    }
}

impl Index2D for (u32, u32) {
    fn to_2d(&self, _: u32) -> (u32, u32) {
        *self
    }
}
