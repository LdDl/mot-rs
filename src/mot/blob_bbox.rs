use uuid::Uuid;

use kalman_rust::kalman::KalmanBBox;

use crate::mot::mot_errors;
use crate::utils::{euclidean_distance, Point, Rect};

/// Chi-squared threshold for 4 DOF at 95% confidence level
/// Ref.: Eq.(70) in <https://github.com/LdDl/kalman-rs>
pub const GATING_THRESHOLD_4DOF_95: f32 = 9.488;

/// Chi-squared threshold for 4 DOF at 99% confidence level
pub const GATING_THRESHOLD_4DOF_99: f32 = 13.277;

/// Tracked object using 8D Kalman filter for bounding box tracking.
/// State vector: [cx, cy, w, h, vx, vy, vw, vh]
///
/// Unlike SimpleBlob which only tracks center position (2D state),
/// BlobBBox tracks the full bounding box including width/height dynamics.
/// This enables:
/// - Better predictions when objects change size (zoom in/out)
/// - Mahalanobis distance for uncertainty-aware data association
/// - Chi-squared gating for outlier rejection
#[derive(Debug, Clone)]
pub struct BlobBBox {
    id: Uuid,
    current_bbox: Rect,
    current_center: Point,
    predicted_next_bbox: Rect,
    track: Vec<Point>,
    max_track_len: usize,
    active: bool,
    no_match_times: usize,
    diagonal: f32,
    tracker: KalmanBBox,
    entity_id: usize,
}

impl BlobBBox {
    /// Creates new BlobBBox with specified time delta and Kalman parameters.
    ///
    /// # Arguments
    /// * `_current_bbox` - Initial bounding box (x, y = top-left corner)
    /// * `dt` - Time delta between frames (e.g., 0.04 for 25fps)
    /// * `std_dev_a` - Standard deviation of acceleration (process noise)
    /// * `std_dev_m_pos` - Standard deviation for position measurement noise
    /// * `std_dev_m_size` - Standard deviation for size measurement noise
    pub fn new_with_params(
        _current_bbox: Rect,
        dt: f32,
        std_dev_a: f32,
        std_dev_m_pos: f32,
        std_dev_m_size: f32,
    ) -> Self {
        let center_x = _current_bbox.x + 0.5 * _current_bbox.width;
        let center_y = _current_bbox.y + 0.5 * _current_bbox.height;
        let _diagonal = f32::sqrt(
            _current_bbox.width * _current_bbox.width
                + _current_bbox.height * _current_bbox.height,
        );
        /* Kalman filter props */
        // Control inputs (acceleration assumptions)
        // For size tracking, typically set to 0 (no expected acceleration in size change)
        let u_cx = 1.0;
        let u_cy = 1.0;
        let u_w = 0.0;
        let u_h = 0.0;
        let kf = KalmanBBox::new_with_state(
            dt,
            u_cx,
            u_cy,
            u_w,
            u_h,
            std_dev_a,
            std_dev_m_pos,
            std_dev_m_pos,
            std_dev_m_size,
            std_dev_m_size,
            center_x,
            center_y,
            _current_bbox.width,
            _current_bbox.height,
        );
        let mut newb = BlobBBox {
            id: Uuid::new_v4(),
            current_bbox: _current_bbox.clone(),
            current_center: Point::new(center_x, center_y),
            predicted_next_bbox: _current_bbox,
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
    /// Creates new BlobBBox with specified time delta.
    /// Uses default Kalman parameters: std_dev_a=2.0, std_dev_m_pos=0.1, std_dev_m_size=0.1
    pub fn new_with_dt(_current_bbox: Rect, dt: f32) -> Self {
        Self::new_with_params(_current_bbox, dt, 2.0, 0.1, 0.1)
    }
    /// Creates new BlobBBox with specified center point and bounding box.
    pub fn new_with_center_dt(_current_center: Point, _current_bbox: Rect, dt: f32) -> Self {
        let _diagonal = f32::sqrt(
            _current_bbox.width * _current_bbox.width
                + _current_bbox.height * _current_bbox.height,
        );
        /* Kalman filter props */
        let u_cx = 1.0;
        let u_cy = 1.0;
        let u_w = 0.0;
        let u_h = 0.0;
        let std_dev_a = 2.0;
        let std_dev_m_pos = 0.1;
        let std_dev_m_size = 0.1;
        let kf = KalmanBBox::new_with_state(
            dt,
            u_cx,
            u_cy,
            u_w,
            u_h,
            std_dev_a,
            std_dev_m_pos,
            std_dev_m_pos,
            std_dev_m_size,
            std_dev_m_size,
            _current_center.x,
            _current_center.y,
            _current_bbox.width,
            _current_bbox.height,
        );
        let mut newb = BlobBBox {
            id: Uuid::new_v4(),
            current_bbox: _current_bbox,
            current_center: _current_center,
            predicted_next_bbox: Rect::default(),
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
    /// Creates new BlobBBox with default dt=1.0
    pub fn new(_current_bbox: Rect) -> Self {
        Self::new_with_dt(_current_bbox, 1.0)
    }

    /// Builder pattern to set entity ID
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
    pub fn track_len(&self) -> usize {
        self.track.len()
    }
    pub fn exists(&self) -> bool {
        self.active
    }
    /// Returns predicted position (cx, cy) without mutating state
    pub fn get_predicted_position_readonly(&self) -> (f32, f32) {
        let (cx, cy, _, _) = self.tracker.get_predicted_state();
        (cx, cy)
    }
    /// Returns predicted state (cx, cy, w, h) without mutating state
    pub fn get_predicted_state_readonly(&self) -> (f32, f32, f32, f32) {
        self.tracker.get_predicted_state()
    }
    /// Returns position uncertainty from covariance matrix
    pub fn get_position_uncertainty(&self) -> f32 {
        self.tracker.get_position_uncertainty()
    }
    /// Returns predicted bounding box without mutating state.
    /// Unlike SimpleBlob, this includes predicted width/height changes.
    pub fn get_predicted_bbox_readonly(&self) -> Rect {
        let (pred_cx, pred_cy, pred_w, pred_h) = self.tracker.get_predicted_state();
        Rect::new(
            pred_cx - pred_w / 2.0,
            pred_cy - pred_h / 2.0,
            pred_w,
            pred_h,
        )
    }
    /// Returns current velocity (vx, vy, vw, vh)
    pub fn get_velocity(&self) -> (f32, f32, f32, f32) {
        self.tracker.get_velocity()
    }
    // Execute Kalman filter's first step but without re-evaluating state vector based on Kalman gain
    pub fn predict_next_position(&mut self) {
        self.tracker.predict();
        let (cx, cy, w, h) = self.tracker.get_state();
        self.predicted_next_bbox = Rect::new(
            cx - w / 2.0,
            cy - h / 2.0,
            w,
            h,
        );
    }
    /// Computes squared Mahalanobis distance between predicted state and measurement.
    /// Lower values indicate better match accounting for uncertainty.
    ///
    /// # Arguments
    /// * `bbox` - Measured bounding box to compare against
    ///
    /// # Returns
    /// Squared Mahalanobis distance, or error if matrix inversion fails
    pub fn mahalanobis_distance_squared(&self, bbox: &Rect) -> Result<f32, mot_errors::TrackerError> {
        let cx = bbox.x + bbox.width / 2.0;
        let cy = bbox.y + bbox.height / 2.0;
        self.tracker
            .mahalanobis_distance_squared(cx, cy, bbox.width, bbox.height)
            .map_err(mot_errors::TrackerError::from)
    }
    /// Computes Mahalanobis distance (square root of squared distance)
    pub fn mahalanobis_distance(&self, bbox: &Rect) -> Result<f32, mot_errors::TrackerError> {
        let cx = bbox.x + bbox.width / 2.0;
        let cy = bbox.y + bbox.height / 2.0;
        self.tracker
            .mahalanobis_distance(cx, cy, bbox.width, bbox.height)
            .map_err(mot_errors::TrackerError::from)
    }
    /// Check if measurement passes chi-squared gating threshold.
    /// Ref.: Eq.(70) in <https://github.com/LdDl/kalman-rs>
    ///
    /// # Arguments
    /// * `bbox` - Measured bounding box
    /// * `threshold` - Chi-squared threshold (e.g., GATING_THRESHOLD_4DOF_95)
    pub fn gating_check(&self, bbox: &Rect, threshold: f32) -> bool {
        let cx = bbox.x + bbox.width / 2.0;
        let cy = bbox.y + bbox.height / 2.0;
        self.tracker.gating_check(cx, cy, bbox.width, bbox.height, threshold)
    }
    // Update blob's position and execute Kalman filter's second step (evaluate state vector based on Kalman gain)
    pub fn update(&mut self, newb: &BlobBBox) -> Result<(), mot_errors::TrackerError> {
        // Extract measurement
        let m_cx = newb.current_center.x;
        let m_cy = newb.current_center.y;
        let m_w = newb.current_bbox.width;
        let m_h = newb.current_bbox.height;

        // Smooth center via Kalman filter.
        self.tracker
            .update(m_cx, m_cy, m_w, m_h)
            .map_err(mot_errors::TrackerError::from)?;

        // Update center and re-evaluate bounding box
        let (state_cx, state_cy, state_w, state_h) = self.tracker.get_state();
        self.current_center = Point::new(state_cx, state_cy);
        self.current_bbox = Rect::new(
            state_cx - state_w / 2.0,
            state_cy - state_h / 2.0,
            state_w,
            state_h,
        );
        // Update remaining properties
        self.diagonal = f32::sqrt(state_w * state_w + state_h * state_h);
        self.entity_id = newb.entity_id;
        self.active = true;
        self.no_match_times = 0;

        // Update track
        self.track.push(self.current_center.clone());
        if self.track.len() > self.max_track_len {
            self.track = self.track[1..].to_vec();
        }

        Ok(())
    }
    pub fn distance_to(&self, b: &BlobBBox) -> f32 {
        euclidean_distance(&self.current_center, &b.current_center)
    }
    pub fn distance_to_predicted(&self, b: &BlobBBox) -> f32 {
        let (pred_x1, pred_y1) = self.get_predicted_position_readonly();
        let (pred_x2, pred_y2) = b.get_predicted_position_readonly();
        let p1 = Point::new(pred_x1, pred_y1);
        let p2 = Point::new(pred_x2, pred_y2);
        euclidean_distance(&p1, &p2)
    }
}

use crate::mot::blob::Blob;
impl Blob for BlobBBox {
    fn get_id(&self) -> Uuid { self.id }
    fn set_id(&mut self, new_id: Uuid) { self.id = new_id }
    fn get_center(&self) -> Point { self.current_center.clone() }
    fn get_bbox(&self) -> Rect { self.current_bbox.clone() }
    fn get_diagonal(&self) -> f32 { self.diagonal }
    fn get_predicted_bbox_readonly(&self) -> Rect { BlobBBox::get_predicted_bbox_readonly(self) }
    fn get_predicted_position_readonly(&self) -> (f32, f32) { BlobBBox::get_predicted_position_readonly(self) }
    fn get_position_uncertainty(&self) -> f32 { self.tracker.get_position_uncertainty() }
    fn get_track(&self) -> &Vec<Point> { &self.track }
    fn track_len(&self) -> usize { self.track.len() }
    fn get_max_track_len(&self) -> usize { self.max_track_len }
    fn set_max_track_len(&mut self, max_track_len: usize) { self.max_track_len = max_track_len }
    fn exists(&self) -> bool { self.active }
    fn activate(&mut self) { self.active = true }
    fn deactivate(&mut self) { self.active = false }
    fn get_no_match_times(&self) -> usize { self.no_match_times }
    fn inc_no_match(&mut self) { self.no_match_times += 1 }
    fn reset_no_match(&mut self) { self.no_match_times = 0 }
    fn get_entity_id(&self) -> usize { self.entity_id }
    fn predict_next_position(&mut self) { BlobBBox::predict_next_position(self) }
    fn update(&mut self, measurement: &Self) -> Result<(), mot_errors::TrackerError> {
        BlobBBox::update(self, measurement)
    }
    fn distance_to(&self, other: &Self) -> f32 { BlobBBox::distance_to(self, other) }
    fn distance_to_predicted(&self, other: &Self) -> f32 { BlobBBox::distance_to_predicted(self, other) }
}

impl BlobBBox {
    // Naive old approach to give an idea what is going on
    // I've saved this method just for retrospective
    pub fn predict_next_position_naive(&mut self, _depth: usize) {
        let track_len = self.track.len();
        let depth = usize::min(_depth, track_len);
        if depth <= 1 {
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
            delta_x /= sum;
            delta_y /= sum;
        }
        let pred_x = self.track[track_len - 1].x + delta_x;
        let pred_y = self.track[track_len - 1].y + delta_y;
        self.predicted_next_bbox = Rect::new(
            pred_x - self.current_bbox.width / 2.0,
            pred_y - self.current_bbox.height / 2.0,
            self.current_bbox.width,
            self.current_bbox.height,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blob_bbox_creation() {
        let bbox = Rect::new(100.0, 50.0, 40.0, 80.0);
        let blob = BlobBBox::new_with_dt(bbox.clone(), 0.04);

        let center = blob.get_center();
        assert!((center.x - 120.0).abs() < 0.001);
        assert!((center.y - 90.0).abs() < 0.001);

        let returned_bbox = blob.get_bbox();
        assert!((returned_bbox.x - 100.0).abs() < 0.001);
        assert!((returned_bbox.width - 40.0).abs() < 0.001);
    }

    #[test]
    fn test_blob_bbox_predict_update() {
        let bbox = Rect::new(100.0, 50.0, 40.0, 80.0);
        let mut blob = BlobBBox::new_with_dt(bbox, 0.04);

        blob.predict_next_position();

        let new_bbox = Rect::new(102.0, 52.0, 41.0, 81.0);
        let measurement = BlobBBox::new_with_dt(new_bbox, 0.04);

        blob.update(&measurement).unwrap();

        let center = blob.get_center();
        assert!((center.x - 122.5).abs() < 5.0);
        assert!((center.y - 92.5).abs() < 5.0);
    }

    #[test]
    fn test_mahalanobis_gating() {
        let bbox = Rect::new(100.0, 50.0, 40.0, 80.0);
        let mut blob = BlobBBox::new_with_dt(bbox, 0.04);

        blob.predict_next_position();

        let close_bbox = Rect::new(101.0, 51.0, 40.0, 80.0);
        assert!(blob.gating_check(&close_bbox, GATING_THRESHOLD_4DOF_95));

        let far_bbox = Rect::new(500.0, 500.0, 100.0, 200.0);
        assert!(!blob.gating_check(&far_bbox, GATING_THRESHOLD_4DOF_95));
    }

    #[test]
    fn test_velocity_tracking() {
        let bbox = Rect::new(100.0, 50.0, 40.0, 80.0);
        let mut blob = BlobBBox::new_with_params(bbox, 0.04, 2.0, 0.1, 0.1);

        for i in 1..20 {
            blob.predict_next_position();
            let new_bbox = Rect::new(
                100.0 + i as f32 * 2.0 - 20.5,
                50.0 + i as f32 * 1.5 - 40.15,
                41.0 + i as f32 * 0.5,
                80.3 + i as f32 * 0.3,
            );
            let measurement = BlobBBox::new_with_dt(new_bbox, 0.04);
            blob.update(&measurement).unwrap();
        }

        let (vx, vy, vw, vh) = blob.get_velocity();
        assert!(vx > 0.0, "Velocity X should be positive");
        assert!(vy > 0.0, "Velocity Y should be positive");
        assert!(vw > 0.0, "Velocity W should be positive");
        assert!(vh > 0.0, "Velocity H should be positive");
    }
}
