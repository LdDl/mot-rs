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
    /// Keeps the most recent points up to this limit and trims existing history immediately. Zero clears the history and disables recording without stopping the Kalman filter.
    fn set_max_track_len(&mut self, max_track_len: usize);
    /* Activation state */
    /// For a stored track, true if it was registered or matched in the latest frame. A missed track is inactive but remains stored until its expiry limit is reached.
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
    /// Distance from this blob's current center to the other blob's next predicted center, without advancing either filter.
    fn distance_to_predicted(&self, other: &Self) -> f32;
}

#[cfg(test)]
mod tests {
    use super::Blob;
    use crate::mot::{BlobBBox, SimpleBlob};
    use crate::utils::Rect;

    fn history<B: Blob>(blob: &B) -> Vec<(f32, f32)> {
        blob.get_track()
            .iter()
            .map(|point| (point.x, point.y))
            .collect()
    }

    fn check_history_limit<B: Blob>(new_blob: fn(Rect, f32) -> B, set_limit: fn(&mut B, usize)) {
        let make = |step| new_blob(Rect::new(100.0 + step as f32, 200.0, 20.0, 30.0), 0.04);
        let mut blob = make(0);
        for step in 1..10 {
            blob.predict_next_position();
            blob.update(&make(step)).unwrap();
        }
        let mut observed = history(&blob);
        let prediction = blob.get_predicted_position_readonly();
        let uncertainty = blob.get_position_uncertainty();
        set_limit(&mut blob, 3);
        assert_eq!(blob.get_max_track_len(), 3);
        assert_eq!(history(&blob), observed[observed.len() - 3..]);
        assert_eq!(blob.get_predicted_position_readonly(), prediction);
        assert_eq!(blob.get_position_uncertainty(), uncertainty);

        let capacity = blob.get_track().capacity();
        let storage = blob.get_track().as_ptr();
        for step in 10..20 {
            blob.predict_next_position();
            blob.update(&make(step)).unwrap();
            let center = blob.get_center();
            observed.push((center.x, center.y));
            assert_eq!(history(&blob), observed[observed.len() - 3..]);
            assert_eq!(blob.get_track().capacity(), capacity);
            assert_eq!(blob.get_track().as_ptr(), storage);
        }

        set_limit(&mut blob, 0);
        assert!(blob.get_track().is_empty());
        let center_before = blob.get_center();
        blob.predict_next_position();
        blob.update(&make(30)).unwrap();
        assert!(blob.get_track().is_empty());
        assert_ne!(blob.get_center().x, center_before.x);

        set_limit(&mut blob, 2);
        assert!(blob.get_track().is_empty());
        let mut recent = Vec::new();
        for step in 31..34 {
            blob.predict_next_position();
            blob.update(&make(step)).unwrap();
            let center = blob.get_center();
            recent.push((center.x, center.y));
        }
        assert_eq!(history(&blob), recent[1..]);
        set_limit(&mut blob, 5);
        assert_eq!(history(&blob), recent[1..]);
        blob.predict_next_position();
        blob.update(&make(34)).unwrap();
        assert_eq!(blob.track_len(), 3);
        let center = blob.get_center();
        set_limit(&mut blob, 1);
        assert_eq!(history(&blob), vec![(center.x, center.y)]);
    }

    #[test]
    fn test_history_limit_simple_blob() {
        check_history_limit(SimpleBlob::new_with_dt, SimpleBlob::set_max_track_len);
        check_history_limit(
            SimpleBlob::new_with_dt,
            <SimpleBlob as Blob>::set_max_track_len,
        );
    }

    #[test]
    fn test_history_limit_bbox() {
        check_history_limit(BlobBBox::new_with_dt, BlobBBox::set_max_track_len);
        check_history_limit(
            BlobBBox::new_with_dt,
            <BlobBBox as Blob>::set_max_track_len,
        );
    }
}
