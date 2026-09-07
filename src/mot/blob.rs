use uuid::Uuid;
use crate::mot::mot_errors::TrackerError;
use crate::utils::{Point, Rect};

/// Common interface for tracked objects (blobs).
/// Enables generic trackers: `IoUTracker<B: Blob>`, `ByteTracker<B: Blob>`
///
/// Implementations:
/// - `SimpleBlob` - uses 4D Kalman (center position only)
/// - `BlobBBox` - uses 8D Kalman (full bbox with size dynamics). FUTURE WORKS!!!
pub trait Blob: Clone {
    /* Identity */
    fn get_id(&self) -> Uuid;
    fn set_id(&mut self, new_id: Uuid);
    /* Position and geometry */
    fn get_center(&self) -> Point;
    fn get_bbox(&self) -> Rect;
    fn get_diagonal(&self) -> f32;
    fn get_predicted_bbox_readonly(&self) -> Rect;
    fn get_predicted_position_readonly(&self) -> (f32, f32);
    fn get_position_uncertainty(&self) -> f32;
    /* Track management */
    fn get_track(&self) -> &Vec<Point>;
    fn track_len(&self) -> usize;
    fn get_max_track_len(&self) -> usize;
    fn set_max_track_len(&mut self, max_track_len: usize);
    /* Activation state */
    fn exists(&self) -> bool;
    fn activate(&mut self);
    fn deactivate(&mut self);
    /* No-match tracking (for object lifecycle) */
    fn get_no_match_times(&self) -> usize;
    fn inc_no_match(&mut self);
    fn reset_no_match(&mut self);
    /// How long the object has been unmatched, in seconds: the cycle times of
    /// the consecutive frames it was missed on, summed. Reset by a match.
    /// Unlike `get_no_match_times` it keeps its meaning when the effective frame
    /// rate changes (frame skipping, a throttled detector, a stalled stream)
    fn get_lost_seconds(&self) -> f32;
    /* Entity ID (for external association) */
    fn get_entity_id(&self) -> usize;
    /* Sampling interval */
    /// Cycle time the underlying Kalman filter is currently built for
    fn get_dt(&self) -> f32;
    /// Rebuilds the filter for a new cycle time. Frames rarely arrive at exactly
    /// the nominal rate, and a discrete filter is only valid for the interval
    /// its transition and process-noise matrices were built for
    fn set_dt(&mut self, dt: f32);
    /* Prediction and update */
    fn predict_next_position(&mut self);
    fn update(&mut self, measurement: &Self) -> Result<(), TrackerError>;
    /* Distance metrics */
    fn distance_to(&self, other: &Self) -> f32;
    fn distance_to_predicted(&self, other: &Self) -> f32;
}
