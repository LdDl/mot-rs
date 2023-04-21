use std::error::Error;

use kalman_rust::kalman::{
    Kalman2D
};

use crate::utils::{
    Rect,
    Point,
    euclidean_distance
};

use chrono::{
    DateTime,
    Utc
};

pub trait Blob {
    fn track_len(&self) -> usize;
    fn exists(&self) -> bool;
    fn inc_no_match(&mut self);
    fn dec_no_match(&mut self);
    fn predict_next_position_naive(&mut self, _depth: usize);
}

pub struct SimpleBlob {
    current_bbox: Rect,
    current_center: Point,
    predicted_next_position: Point,
    track: Vec<Point>,
    max_track_len: usize,
    active: bool,
    no_match_times: usize,
    diagonal: f32,
    tracker: Kalman2D
    // @todo: keep track of object timestamps? default/new_with_time(...)?
}

impl SimpleBlob {
    pub fn new(_current_bbox: Rect) -> Self {
        let center_x = _current_bbox.x as f32 + 0.5 * _current_bbox.width as f32;
        let center_y = _current_bbox.y as f32 + 0.5 * _current_bbox.height as f32;
        let _diagonal = f32::sqrt((_current_bbox.width*_current_bbox.width) as f32 + (_current_bbox.height*_current_bbox.height) as f32);

        /* Kalman filter props */
        //
        // Why set initial state at all? See answer here: https://github.com/LdDl/kalman-rs/blob/master/src/kalman/kalman_2d.rs#L126
        //
        let dt = 1.0;
        let ux = 1.0;
        let uy = 1.0;
        let std_dev_a = 2.0;
        let std_dev_mx = 0.1;
        let std_dev_my = 0.1;
        let kf = Kalman2D::new_with_state(dt, ux, uy, std_dev_a, std_dev_mx, std_dev_my, center_x, center_y);
        SimpleBlob {
            current_bbox: _current_bbox,
            current_center: Point::new(f32::round(center_x) as i32, f32::round(center_y) as i32),
            predicted_next_position: Point::default(),
            track: Vec::new(),
            max_track_len: 150,
            active: false,
            no_match_times: 0,
            diagonal: _diagonal,
            tracker: kf
        }
    }
    pub fn partial_copy(&self) -> Self {
        let dt = 1.0;
        let ux = 1.0;
        let uy = 1.0;
        let std_dev_a = 2.0;
        let std_dev_mx = 0.1;
        let std_dev_my = 0.1;
        let kf = Kalman2D::new_with_state(dt, ux, uy, std_dev_a, std_dev_mx, std_dev_my, self.current_center.x as f32, self.current_center.y as f32);
        SimpleBlob {
            current_bbox: self.current_bbox.clone(),
            current_center: self.current_center.clone(),
            predicted_next_position: Point::default(),
            track: Vec::new(),
            max_track_len: self.max_track_len,
            active: false,
            no_match_times: 0,
            diagonal: self.diagonal,
            tracker: kf
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
    pub fn get_max_track_len(&self) -> usize{
        self.max_track_len
    }
    pub fn set_max_track_len(&mut self, max_track_len: usize) {
        self.max_track_len = max_track_len
    }
    pub fn get_no_match_times(&self) -> usize {
        self.no_match_times
    }
    pub fn inc_no_match(&mut self) {
        self.no_match_times += 1
    }
    // Execute Kalman filter's first step but without re-evaluating state vector based on Kalman gain
    pub fn predict_next_position(&mut self) {
        self.tracker.predict();
        let (state_x, state_y) = self.tracker.get_state();
        self.predicted_next_position.x = f32::round(state_x) as i32;
        self.predicted_next_position.y = f32::round(state_y) as i32;
    }
    // Naive old approach to give an idea what is going on
    // I've saved this method just for retrospective
    pub fn predict_next_position_naive(&mut self, _depth: usize) {
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
    // Update blobs position and execute Kalman filter's second step (evalute state vector based on Kalman gain)
    pub fn update(&mut self, newb: &SimpleBlob) -> Result<(), Box<dyn Error>> {
        // Update center
        self.current_center = newb.current_center.to_owned();
        self.current_bbox = newb.current_bbox.to_owned();

        // Smooth center via Kalman filter.
        match self.tracker.update(self.current_center.x as f32, self.current_center.y as f32) {
            Ok(_) =>{
                // Update center and re-evaluate bounding box
                let (state_x, state_y) = self.tracker.get_state();
                let old_x = self.current_center.x;
                let old_y = self.current_center.y;
                self.current_center.x = f32::round(state_x) as i32;
                self.current_center.y = f32::round(state_y) as i32;
                let diff_x = self.current_center.x - old_x;
                let diff_y = self.current_center.y - old_y;
                self.current_bbox = Rect::new(
                    self.current_bbox.x - diff_x,
                    self.current_bbox.y - diff_y,
                    self.current_bbox.width - diff_x,
                    self.current_bbox.height - diff_y
                );
                // Update remaining properties
                self.diagonal = newb.diagonal;
                self.active = true;
                self.no_match_times = 0;

                // Update track
                self.track.push(self.current_center.clone());
                if self.track.len() > self.max_track_len {
                    self.track = self.track[1..].to_vec();
                }
            },
            Err(e) => {
                return Err(format!("Can't update object tracker: {}", e))?;
            }
        };
        Ok(())
    }
    pub fn distance_to(&self, b: &SimpleBlob) -> f32 {
        euclidean_distance(&self.current_center, &b.current_center)
    }
    pub fn distance_to_predicted(&self, b: &SimpleBlob) -> f32 {
        euclidean_distance(&self.predicted_next_position, &b.predicted_next_position)
    }
}