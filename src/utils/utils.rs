#[derive(Clone)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32
}

impl Rect {
    pub fn default() -> Rect {
        Rect {
            x: 0,
            y: 0,
            width: 0,
            height: 0
        }
    }
    pub fn new(_x: i32, _y: i32, _height: i32, _width: i32) -> Rect {
        Rect {
            x: _x,
            y: _y,
            width: _height,
            height: _width
        }
    }
}

#[derive(Clone)]
pub struct Point {
    pub x: i32,
    pub y: i32
}

impl Point {
    pub fn default() -> Point {
        Point {
            x: 0,
            y: 0
        }
    }
}

pub fn euclidean_distance(p1: &Point, p2: &Point) -> f32 {
    let x_squared = i32::pow(i32::abs(p1.x - p2.x), 2);
    let y_squared = i32::pow(i32::abs(p1.y - p2.y), 2);
    let sum_f32 = (x_squared + y_squared) as f32;
    return f32::sqrt(sum_f32)
}

mod tests {
    use super::*;
    #[test]
    fn test_euclidean_distance() {
        let p1 = Point{x: 341, y: 264};
        let p2 = Point{x: 421, y: 427};
        let ans = euclidean_distance(&p1, &p2);
        assert_eq!(181.57367651, ans);
    }
}

