#[derive(Clone, Default)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32
}

impl Rect {
    pub fn new(_x: i32, _y: i32, _height: i32, _width: i32) -> Self {
        Rect {
            x: _x,
            y: _y,
            width: _height,
            height: _width
        }
    }
}

#[derive(Clone, Default)]
pub struct Point {
    pub x: i32,
    pub y: i32
}

impl Point {
    pub fn new(_x: i32, _y: i32) -> Self {
        Point {
            x: _x,
            y: _y
        }
    }
}

pub fn euclidean_distance(p1: &Point, p2: &Point) -> f32 {
    let x_squared = i32::pow(i32::abs(p1.x - p2.x), 2);
    let y_squared = i32::pow(i32::abs(p1.y - p2.y), 2);
    let sum_f32 = (x_squared + y_squared) as f32;
    f32::sqrt(sum_f32)
}

mod tests {
    #[test]
    fn test_euclidean_distance() {
        let p1 = super::Point::new(341, 264);
        let p2 = super::Point::new(421, 427);
        let ans = super::euclidean_distance(&p1, &p2);
        assert_eq!(181.57367, ans);
    }
}

