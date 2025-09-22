use uuid::Uuid;

use kalman_rust::kalman::Kalman2D;

use crate::mot::mot_errors;
use crate::utils::{euclidean_distance, Point, Rect};

pub trait Blob {
    fn track_len(&self) -> usize;
    fn exists(&self) -> bool;
    fn inc_no_match(&mut self);
    fn dec_no_match(&mut self);
    fn predict_next_position_naive(&mut self, _depth: usize);
}

#[derive(Debug, Clone)]
pub struct SimpleBlob {
    id: Uuid,
    current_bbox: Rect,
    current_center: Point,
    predicted_next_position: Point,
    track: Vec<Point>,
    max_track_len: usize,
    active: bool,
    no_match_times: usize,
    diagonal: f32,
    tracker: Kalman2D,
    entity_id: usize,
    // @todo: keep track of object timestamps? default/new_with_time(...)?
}

impl SimpleBlob {
    pub fn new_with_dt(_current_bbox: Rect, dt: f32) -> Self {
        let center_x = _current_bbox.x as f32 + 0.5 * _current_bbox.width as f32;
        let center_y = _current_bbox.y as f32 + 0.5 * _current_bbox.height as f32;
        let _diagonal = f32::sqrt(
            (_current_bbox.width * _current_bbox.width) as f32
                + (_current_bbox.height * _current_bbox.height) as f32,
        );
        /* Kalman filter props */
        //
        // Why set initial state at all? See answer here: https://github.com/LdDl/kalman-rs/blob/master/src/kalman/kalman_2d.rs#L126
        //
        let ux = 1.0;
        let uy = 1.0;
        let std_dev_a = 2.0;
        let std_dev_mx = 0.1;
        let std_dev_my = 0.1;
        let kf = Kalman2D::new_with_state(
            dt, ux, uy, std_dev_a, std_dev_mx, std_dev_my, center_x, center_y,
        );
        let mut newb = SimpleBlob {
            id: Uuid::new_v4(),
            current_bbox: _current_bbox,
            current_center: Point::new(center_x, center_y),
            predicted_next_position: Point::default(),
            track: Vec::with_capacity(150),
            max_track_len: 150,
            active: false,
            no_match_times: 0,
            diagonal: _diagonal,
            tracker: kf,
            entity_id: 0,
        };
        newb.track.push(newb.current_center.clone());
        newb
    }
    pub fn new_with_center_dt(_current_center: Point, _current_bbox: Rect, dt: f32) -> Self {
        let _diagonal = f32::sqrt(
            (_current_bbox.width * _current_bbox.width) as f32
                + (_current_bbox.height * _current_bbox.height) as f32,
        );
        /* Kalman filter props */
        //
        // Why set initial state at all? See answer here: https://github.com/LdDl/kalman-rs/blob/master/src/kalman/kalman_2d.rs#L126
        //
        let ux = 1.0;
        let uy = 1.0;
        let std_dev_a = 2.0;
        let std_dev_mx = 0.1;
        let std_dev_my = 0.1;
        let kf = Kalman2D::new_with_state(
            dt,
            ux,
            uy,
            std_dev_a,
            std_dev_mx,
            std_dev_my,
            _current_center.x,
            _current_center.y,
        );
        let mut newb = SimpleBlob {
            id: Uuid::new_v4(),
            current_bbox: _current_bbox,
            current_center: _current_center,
            predicted_next_position: Point::default(),
            track: Vec::with_capacity(150),
            max_track_len: 150,
            active: false,
            no_match_times: 0,
            diagonal: _diagonal,
            tracker: kf,
            entity_id: 0,
        };
        newb.track.push(newb.current_center.clone());
        newb
    }
    pub fn new(_current_bbox: Rect) -> Self {
        return SimpleBlob::new_with_dt(_current_bbox, 1.0);
    }

    /// Add some entity_id be able to reassign a tracking uuid to an entity
    pub fn with_entity_id(mut self, eid: usize) -> Self {
        self.entity_id = eid;
        self
    }

    pub fn get_entity_id(&self) -> usize {
        self.entity_id
    }

    pub fn activate(&mut self) {
        self.active = true
    }
    pub fn deactivate(&mut self) {
        self.active = false
    }
    pub fn get_id(&self) -> Uuid {
        self.id
    }
    pub fn set_id(&mut self, new_id: Uuid) {
        self.id = new_id
    }
    pub fn get_center(&self) -> Point {
        self.current_center.clone()
    }
    pub fn get_bbox(&self) -> Rect {
        self.current_bbox.clone()
    }
    pub fn get_diagonal(&self) -> f32 {
        self.diagonal
    }
    pub fn get_track(&self) -> &Vec<Point> {
        &self.track
    }
    pub fn get_max_track_len(&self) -> usize {
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
    pub fn reset_no_match(&mut self) {
        self.no_match_times = 0
    }
    /// Returns predicted position without mutating state (read-only peek)
    pub fn get_predicted_position_readonly(&self) -> (f32, f32) {
        self.tracker.get_predicted_position()
    }
    /// Returns position uncertainty for data association
    pub fn get_position_uncertainty(&self) -> f32 {
        self.tracker.get_position_uncertainty()
    }
    /// Returns predicted bounding box without mutating state
    pub fn get_predicted_bbox_readonly(&self) -> Rect {
        let (pred_x, pred_y) = self.get_predicted_position_readonly();
        let diff_x = pred_x - self.current_center.x;
        let diff_y = pred_y - self.current_center.y;

        Rect::new(
            self.current_bbox.x + diff_x,
            self.current_bbox.y + diff_y,
            self.current_bbox.width,
            self.current_bbox.height,
        )
    }
    // Execute Kalman filter's first step but without re-evaluating state vector based on Kalman gain
    pub fn predict_next_position(&mut self) {
        self.tracker.predict();
        let (state_x, state_y) = self.tracker.get_state();
        self.predicted_next_position.x = state_x;
        self.predicted_next_position.y = state_y;
    }
    // Naive old approach to give an idea what is going on
    // I've saved this method just for retrospective
    pub fn predict_next_position_naive(&mut self, _depth: usize) {
        let track_len = self.track.len();
        let depth = usize::min(_depth, track_len);
        if depth <= 1 {
            self.predicted_next_position.x = 0.0;
            self.predicted_next_position.y = 0.0;
            return;
        }
        let mut current = track_len - 1;
        let mut prev = current - 1;
        let mut delta_x = 0.0;
        let mut delta_y = 0.0;
        let mut sum = 0.0;
        for i in 1..depth {
            let weight = (depth - i) as f32;
            delta_x += (self.track[current].x - self.track[prev].x) * weight;
            delta_y += (self.track[current].y - self.track[prev].y) * weight;
            sum += i as f32;
            current = prev;
            if current != 0 {
                prev = current - 1;
            }
        }
        if sum > 0.0 {
            delta_x = delta_x / sum;
            delta_y = delta_y / sum;
        }
        self.predicted_next_position.x = self.track[track_len - 1].x + delta_x;
        self.predicted_next_position.y = self.track[track_len - 1].y + delta_y;
    }
    // Update blobs position and execute Kalman filter's second step (evalute state vector based on Kalman gain)
    pub fn update(&mut self, newb: &SimpleBlob) -> Result<(), mot_errors::TrackerError> {
        // Update center
        self.current_center = newb.current_center.to_owned();
        self.current_bbox = newb.current_bbox.to_owned();
        self.entity_id = newb.entity_id;

        // Smooth center via Kalman filter.
        self.tracker
            .update(newb.current_center.x as f32, newb.current_center.y as f32)?;

        // Update center and re-evaluate bounding box
        let (state_x, state_y) = self.tracker.get_state();
        let old_x = self.current_center.x;
        let old_y = self.current_center.y;
        self.current_center.x = state_x;
        self.current_center.y = state_y;
        let diff_x = self.current_center.x - old_x;
        let diff_y = self.current_center.y - old_y;
        self.current_bbox = Rect::new(
            self.current_bbox.x - diff_x,
            self.current_bbox.y - diff_y,
            self.current_bbox.width - diff_x,
            self.current_bbox.height - diff_y,
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

        Ok(())
    }
    pub fn distance_to(&self, b: &SimpleBlob) -> f32 {
        euclidean_distance(&self.current_center, &b.current_center)
    }
    pub fn distance_to_predicted(&self, b: &SimpleBlob) -> f32 {
        euclidean_distance(&self.predicted_next_position, &b.predicted_next_position)
    }
}
