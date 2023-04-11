use crate::utils::{
    Rect,
    Point,
    euclidean_distance
};

pub trait Blob {
    fn track_len(&self) -> usize;
    fn exists(&self) -> bool;
    fn inc_no_match(&mut self);
    fn dec_no_match(&mut self);
    fn predict_next_position(&mut self, _depth: usize);
}

pub struct SimpleBlob {
    current_bbox: Rect,
    current_center: Point,
    predicted_next_position: Point,
    track: Vec<Point>,
    active: bool,
    no_match_times: usize,
    diagonal: f32
}

impl SimpleBlob {
    pub fn new(_current_bbox: Rect) -> Self {
        let center_x = _current_bbox.x as f32 + 0.5 * _current_bbox.width as f32;
        let center_y = _current_bbox.y as f32 + 0.5 * _current_bbox.height as f32;
        let _diagonal = f32::sqrt((_current_bbox.width*_current_bbox.width) as f32 + (_current_bbox.height*_current_bbox.height) as f32);
        SimpleBlob {
            current_bbox: _current_bbox,
            current_center: Point{x: f32::round(center_x) as i32, y: f32::round(center_y) as i32},
            predicted_next_position: Point::default(),
            track: Vec::new(),
            active: false,
            no_match_times: 0,
            diagonal: _diagonal
        }
    }
    pub fn partial_copy(&self) -> Self {
        SimpleBlob {
            current_bbox: self.current_bbox.clone(),
            current_center: self.current_center.clone(),
            predicted_next_position: Point::default(),
            track: Vec::new(),
            active: false,
            no_match_times: 0,
            diagonal: self.diagonal
        }
    }
    pub fn activate(&mut self) {
        self.active = true
    }
    pub fn deactivate(&mut self) {
        self.active = false
    }
    pub fn get_diagonal(&self) -> f32 {
        self.diagonal
    }
    pub fn inc_no_match(&mut self) {
        self.no_match_times += 1
    }
    fn dec_no_match(&mut self) {
        self.no_match_times -= 1
    }
    pub fn predict_next_position(&mut self, _depth: usize) {
        let track_len = self.track.len();
        let depth = usize::min(_depth, track_len);
        if depth <= 1 {
            self.predicted_next_position.x = 0;
            self.predicted_next_position.y = 0;
            return
        }
        let mut current = track_len - 1;
        let mut prev = current - 1;
        let mut delta_x = 0;
        let mut delta_y = 0;
        let mut sum = 0;
        for i in 1..depth {
            let weight = (depth - i) as i32;
            delta_x += (self.track[current].x - self.track[prev].x) * weight;
		    delta_y += (self.track[current].y - self.track[prev].y) * weight;
            sum += i as i32;
            current = prev;
            if current != 0 {
                prev = current - 1;
            }
        }
        if sum > 0 {
            delta_x = f32::round(delta_x as f32 / sum as f32) as i32;
            delta_y = f32::round(delta_y as f32 / sum as f32) as i32;
        }
        self.predicted_next_position.x = self.track[track_len - 1].x + delta_x;
        self.predicted_next_position.y = self.track[track_len - 1].y + delta_y;
    }
    pub fn update(&mut self, newb: &SimpleBlob) {
        //@todo update state
    }

    pub fn distance_to(&self, b: &SimpleBlob) -> f32 {
        return euclidean_distance(&self.current_center, &b.current_center);
    }
    pub fn distance_to_predicted(&self, b: &SimpleBlob) -> f32 {
        return euclidean_distance(&self.predicted_next_position, &b.predicted_next_position);
    }
}