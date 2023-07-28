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

pub fn iou(r1: &Rect, r2: &Rect) -> f32 {
    let x1 = r1.x;
    let y1 = r1.y;
    let x2 = r1.x + r1.width;
    let y2 = r1.y + r1.height;

    let x3 = r2.x;
    let y3 = r2.y;
    let x4 = r2.x + r2.width;
    let y4 = r2.y + r2.height;

    let x_overlap_width = f32::max(0.0, f32::min(x2, x4) - f32::max(x1, x3));
    let y_overlap_height = f32::max(0.0, f32::min(y2, y4) - f32::max(y1, y3));

    if x_overlap_width <= 0.0 || y_overlap_height <= 0.0 {
        return 0.0;
    }
    
    let intersection_area = x_overlap_width * y_overlap_height;
    let r1_area = r1.width * r1.height;
    let r2_area = r2.width * r2.height;
    let union_area = r1_area + r2_area - intersection_area;

    if union_area <= 0.0 {
        return 0.0;
    }

    let iou = intersection_area / union_area;

    f32::max(0.0, f32::min(iou, 1.0))
}

mod tests {
    #[test]
    fn test_euclidean_distance() {
        let p1 = super::Point::new(341.0, 264.0);
        let p2 = super::Point::new(421.0, 427.0);
        let ans = super::euclidean_distance(&p1, &p2);
        assert_eq!(181.57367, ans);
    }

    #[test]
    fn test_iou() {
        // Test cases with known IoU values for validation
        let rect1 = super::Rect::new(0.0, 0.0, 10.0, 10.0);
        let rect2 = super::Rect::new(5.0, 5.0, 10.0, 10.0);
        assert_eq!(super::iou(&rect1, &rect2), 0.14285715);

        let rect3 = super::Rect::new(10.0, 10.0, 10.0, 10.0);
        let rect4 = super::Rect::new(20.0, 20.0, 10.0, 10.0);
        assert_eq!(super::iou(&rect3, &rect4), 0.0);

        let rect5 = super::Rect::new(0.0, 0.0, 20.0, 20.0);
        let rect6 = super::Rect::new(5.0, 5.0, 10.0, 10.0);
        assert_eq!(super::iou(&rect5, &rect6), 0.25);

        let rect7 = super::Rect::new(0.0, 0.0, 10.0, 10.0);
        let rect8 = super::Rect::new(0.0, 0.0, 10.0, 10.0);
        assert_eq!(super::iou(&rect7, &rect8), 1.0);

        // Test case with two rectangles having zero width and height
        let rect9 = super::Rect::new(0.0, 0.0, 0.0, 0.0);
        let rect10 = super::Rect::new(5.0, 5.0, 0.0, 0.0);
        assert_eq!(super::iou(&rect9, &rect10), 0.0);

        // Test case with two rectangles are the same
        let rect11 = super::Rect::new(4.5, 2.0, 10.0, 10.0);
        let rect12 = super::Rect::new(4.5, 2.0, 10.0, 10.0);
        assert_eq!(super::iou(&rect11, &rect12), 1.0);
    }
}

