#[derive(Clone, Default, Debug)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32
}

impl Rect {
    pub fn new(_x: f32, _y: f32, _width: f32, _height: f32) -> Self {
        Rect {
            x: _x,
            y: _y,
            width: _width,
            height: _height
        }
    }
}

#[derive(Clone, Default, Debug)]
pub struct Point {
    pub x: f32,
    pub y: f32
}

impl Point {
    pub fn new(_x: f32, _y: f32) -> Self {
        Point {
            x: _x,
            y: _y
        }
    }
}

pub fn euclidean_distance(p1: &Point, p2: &Point) -> f32 {
    let x_squared = f32::powf(f32::abs(p1.x - p2.x), 2.0);
    let y_squared = f32::powf(f32::abs(p1.y - p2.y), 2.0);
    let sum_f32 = (x_squared + y_squared) as f32;
    f32::sqrt(sum_f32)
}

mod tests {
    #[test]
    fn test_euclidean_distance() {
        let p1 = super::Point::new(341.0, 264.0);
        let p2 = super::Point::new(421.0, 427.0);
        let ans = super::euclidean_distance(&p1, &p2);
        assert_eq!(181.57367, ans);
    }
}

