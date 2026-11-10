use std::collections::{HashMap, HashSet};

use crate::mot::blob::Blob;
use crate::mot::greedy_matching::greedy_matching;
use crate::mot::mot_errors;
use uuid::Uuid;

/// Naive implementation of Multi-object tracker (MOT)
pub struct SimpleTracker<B: Blob> {
    // Max no match (max number of frames when object could not be found again). Default is 75
    max_no_match: usize,
    // When set, tracks expire by unmatched time instead of by `max_no_match` frames. Default is None
    max_lost_seconds: Option<f32>,
    // Threshold distance (most of time in pixels). Default 30.0
    min_dist_threshold: f32,
    // Storage
    pub objects: HashMap<Uuid, B>,
}

impl<B: Blob> SimpleTracker<B> {
    /// Creates default instance of SimpleTracker
    ///
    /// Basic usage:
    ///
    /// ```
    /// use mot_rs::mot::{SimpleTracker, SimpleBlob};
    /// let mut tracker: SimpleTracker<SimpleBlob> = SimpleTracker::default();
    /// ```
    pub fn default() -> Self {
        SimpleTracker {
            max_no_match: 75,
            max_lost_seconds: None,
            min_dist_threshold: 30.0,
            objects: HashMap::new(),
        }
    }
    /// Creates news instance of SimpleTracker
    ///
    /// Basic usage:
    ///
    /// ```
    /// use mot_rs::mot::{SimpleTracker, SimpleBlob};
    /// let max_no_match: usize = 100;
    /// let min_dist_threshold: f32 = 15.0;
    /// let mut tracker: SimpleTracker<SimpleBlob> = SimpleTracker::new(max_no_match, min_dist_threshold);
    /// ```
    pub fn new(_max_no_match: usize, _min_dist_threshold: f32) -> Self {
        SimpleTracker {
            max_no_match: _max_no_match,
            max_lost_seconds: None,
            min_dist_threshold: _min_dist_threshold,
            objects: HashMap::new(),
        }
    }
    /// Switches track expiry from a frame count to time: a track is removed once
    /// it has been unmatched for more than `seconds`. A frame count changes
    /// meaning whenever the effective frame rate does - frame skipping, a
    /// throttled detector, a stalled stream - while an occlusion lasts the same
    /// number of seconds regardless. Non-positive values are ignored
    pub fn set_max_lost_seconds(&mut self, seconds: f32) {
        if seconds > 0.0 {
            self.max_lost_seconds = Some(seconds);
        }
    }
    /// Switches track expiry back to a frame count (an object is removed after more than `max_no_match` missed frames)
    pub fn set_max_no_match(&mut self, max_no_match: usize) {
        self.max_no_match = max_no_match;
        self.max_lost_seconds = None;
    }
    /// Time-based expiry limit, if enabled
    pub fn get_max_lost_seconds(&self) -> Option<f32> {
        self.max_lost_seconds
    }
    /// Frame-based expiry limit; in effect only while `get_max_lost_seconds` is `None`
    pub fn get_max_no_match(&self) -> usize {
        self.max_no_match
    }
    /// Rebuilds every track for a new cycle time. `match_objects` does the same
    /// from the first detection it receives, but a frame without detections
    /// carries none, so call this before it: otherwise on such frames the
    /// tracks are predicted and expired over whatever interval the previous
    /// frame had, not the real one
    pub fn set_dt(&mut self, dt: f32) {
        for (_, object) in self.objects.iter_mut() {
            object.set_dt(dt);
        }
    }

    // Matches new objects to existing ones
    pub fn match_objects(
        &mut self,
        new_objects: &mut Vec<B>,
    ) -> Result<(), mot_errors::TrackerError> {
        // Use the current frame interval for both prediction and lost time.
        if let Some(first) = new_objects.first() {
            self.set_dt(first.get_dt());
        }

        for (_, object) in self.objects.iter_mut() {
            object.deactivate();
        }
        let tracks: Vec<_> = if new_objects.is_empty() {
            Vec::new()
        } else {
            self.objects.iter().collect()
        };
        let candidates = new_objects
            .iter()
            .map(|detection| {
                let mut row = Vec::new();
                let diagonal_threshold = detection.get_diagonal() * 0.5;
                for (index, (_, object)) in tracks.iter().enumerate() {
                    // Preserve custom Blob distance metrics and evaluate each pair only once.
                    let distance = f32::min(
                        detection.distance_to(object),
                        detection.distance_to_predicted(object),
                    );
                    if distance < f32::MAX
                        && (distance < diagonal_threshold || distance < self.min_dist_threshold)
                    {
                        row.push((index, distance));
                    }
                }
                row
            })
            .collect();
        let matches = greedy_matching(candidates, tracks.len());
        let track_ids: Vec<_> = tracks.into_iter().map(|(id, _)| *id).collect();
        let mut reserved_objects = HashSet::with_capacity(matches.len());
        let mut matched_detections = vec![false; new_objects.len()];
        for (detection_index, track_index) in matches {
            let id = track_ids[track_index];
            let track = self.objects.get_mut(&id).ok_or_else(|| {
                mot_errors::TrackerError::from(mot_errors::NoObjectInTracker {
                    txt: format!("Object ID {:?} is missing from tracker", id),
                })
            })?;
            let detection = &mut new_objects[detection_index];
            track.predict_next_position();
            track.update(detection)?;
            track.activate();
            track.reset_no_match();
            detection.set_id(id);
            reserved_objects.insert(id);
            matched_detections[detection_index] = true;
        }

        let mut blobs_to_register = HashMap::new();
        for (index, detection) in new_objects.iter().enumerate() {
            if !matched_detections[index] {
                let mut track = detection.clone();
                track.activate();
                blobs_to_register.insert(track.get_id(), track);
            }
        }

        // Count misses only for existing tracks without a detection this frame.
        for (id, object) in self.objects.iter_mut() {
            if !reserved_objects.contains(id) {
                object.predict_next_position();
                object.inc_no_match();
            }
        }

        self.objects.extend(blobs_to_register);

        // Clean up existing data
        let max_no_match = self.max_no_match;
        let max_lost_seconds = self.max_lost_seconds;
        self.objects.retain(|_, object| {
            // Remove object if it was not found for a long time
            let delete = match max_lost_seconds {
                Some(seconds) => object.get_lost_seconds() > seconds,
                None => object.get_no_match_times() > max_no_match,
            };
            !delete // <- if we want to keep object closure should return true
        });
        Ok(())
    }
}

use std::fmt;
impl<B: Blob> fmt::Display for SimpleTracker<B> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Maximum no match: {}\n\tMinimum threshold distance: {}",
            self.max_no_match, self.min_dist_threshold
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::mot::blob::Blob;
    use crate::mot::test_data::{bbox_to_rect, get_naive_data, get_spread_data};
    use crate::mot::{BlobBBox, SimpleBlob};
    use crate::utils::{euclidean_distance, Point, Rect};

    fn check_activity_follows_detections<B: Blob>(new_blob: fn(Rect, f32) -> B) {
        let mut tracker = super::SimpleTracker::new(5, 30.0);
        let make = |x| new_blob(Rect::new(x, 100.0, 100.0, 100.0), 0.04);
        let mut frame = vec![make(100.0), make(1000.0)];
        let first_id = frame[0].get_id();
        let second_id = frame[1].get_id();
        tracker.match_objects(&mut frame).unwrap();
        assert!(tracker.objects[&first_id].exists());
        assert!(tracker.objects[&second_id].exists());

        let mut frame = vec![make(101.0), make(2000.0)];
        let new_id = frame[1].get_id();
        tracker.match_objects(&mut frame).unwrap();
        assert_eq!(frame[0].get_id(), first_id);
        assert!(tracker.objects[&first_id].exists());
        assert!(!tracker.objects[&second_id].exists());
        assert!(tracker.objects[&new_id].exists());
        assert_eq!(tracker.objects[&second_id].get_no_match_times(), 1);

        tracker.match_objects(&mut vec![]).unwrap();
        assert_eq!(tracker.objects.len(), 3);
        assert!(tracker.objects.values().all(|track| !track.exists()));

        let mut frame = vec![make(1000.0)];
        tracker.match_objects(&mut frame).unwrap();
        assert_eq!(frame[0].get_id(), second_id);
        assert!(tracker.objects[&second_id].exists());
        assert_eq!(tracker.objects[&second_id].get_no_match_times(), 0);
        assert_eq!(tracker.objects[&second_id].get_lost_seconds(), 0.0);
        assert!(!tracker.objects[&first_id].exists());
        assert!(!tracker.objects[&new_id].exists());
    }

    #[test]
    fn test_activity_follows_detections() {
        check_activity_follows_detections(SimpleBlob::new_with_dt);
        check_activity_follows_detections(BlobBBox::new_with_dt);
    }

    fn check_competing_detections<B: Blob>(
        new_blob: fn(Rect, f32) -> B,
        track_positions: &[f32],
        detections: &[(f32, Option<usize>)],
    ) {
        use itertools::Itertools;
        use std::collections::HashSet;

        for detections in detections.iter().permutations(detections.len()) {
            let dt = 0.04;
            let mut tracker = super::SimpleTracker::new(5, 30.0);
            let mut frame: Vec<B> = track_positions
                .iter()
                .map(|&x| new_blob(Rect::new(x, 100.0, 100.0, 100.0), dt))
                .collect();
            let track_ids: Vec<_> = frame.iter().map(Blob::get_id).collect();
            tracker.match_objects(&mut frame).unwrap();

            let mut frame: Vec<B> = detections
                .iter()
                .map(|&&(x, _)| new_blob(Rect::new(x, 100.0, 100.0, 100.0), dt))
                .collect();
            let detection_ids: Vec<_> = frame.iter().map(Blob::get_id).collect();
            tracker.match_objects(&mut frame).unwrap();

            let assigned_ids: HashSet<_> = frame.iter().map(Blob::get_id).collect();
            assert_eq!(assigned_ids.len(), frame.len());
            let new_tracks = detections.iter().filter(|(_, index)| index.is_none()).count();
            assert_eq!(tracker.objects.len(), track_ids.len() + new_tracks);
            for (i, (_, track_index)) in detections.iter().enumerate() {
                let expected_id = track_index.map_or(detection_ids[i], |index| track_ids[index]);
                assert_eq!(frame[i].get_id(), expected_id, "detections: {detections:?}");
                let track = &tracker.objects[&expected_id];
                assert_eq!(track.get_no_match_times(), 0);
                assert_eq!(track.get_lost_seconds(), 0.0);
                assert_eq!(track.track_len(), if track_index.is_some() { 2 } else { 1 });
            }
            for (index, id) in track_ids.iter().enumerate() {
                if !detections.iter().any(|(_, target)| *target == Some(index)) {
                    let track = &tracker.objects[id];
                    assert_eq!(track.get_no_match_times(), 1);
                    assert_eq!(track.get_lost_seconds(), dt);
                    assert_eq!(track.track_len(), 1);
                }
            }
        }
    }

    #[test]
    fn test_competing_detections_use_free_track() {
        let detections = [(101.0, Some(0)), (110.0, Some(1))];
        check_competing_detections(SimpleBlob::new_with_dt, &[100.0, 130.0], &detections);
        check_competing_detections(BlobBBox::new_with_dt, &[100.0, 130.0], &detections);
    }

    #[test]
    fn test_competing_detections_preserve_match_priority() {
        // The detection at 118 wins the second track before the one at 110 retries the third.
        let detections = [(101.0, Some(0)), (110.0, Some(2)), (118.0, Some(1))];
        let tracks = [100.0, 130.0, 160.0];
        check_competing_detections(SimpleBlob::new_with_dt, &tracks, &detections);
        check_competing_detections(BlobBBox::new_with_dt, &tracks, &detections);
    }

    #[test]
    fn test_competing_detections_register_when_no_suitable_track_remains() {
        let detections = [(101.0, Some(0)), (110.0, None)];
        for tracks in [&[100.0][..], &[100.0, 1000.0][..]] {
            check_competing_detections(SimpleBlob::new_with_dt, tracks, &detections);
            check_competing_detections(BlobBBox::new_with_dt, tracks, &detections);
        }
    }

    fn check_prediction_distance_uses_detection_center<B: Blob>(new_blob: fn(Rect, f32) -> B) {
        let track = new_blob(Rect::new(10.0, 20.0, 10.0, 10.0), 1.0);
        let detection = new_blob(Rect::new(100.0, 20.0, 10.0, 10.0), 1.0);
        let prediction = track.get_predicted_position_readonly();
        let detection_prediction = detection.get_predicted_position_readonly();
        let track_uncertainty = track.get_position_uncertainty();
        let detection_uncertainty = detection.get_position_uncertainty();
        let expected = euclidean_distance(
            &detection.get_center(),
            &Point::new(prediction.0, prediction.1),
        );

        assert_eq!(detection.distance_to_predicted(&track), expected);
        assert_eq!(track.get_predicted_position_readonly(), prediction);
        assert_eq!(
            detection.get_predicted_position_readonly(),
            detection_prediction
        );
        assert_eq!(track.get_position_uncertainty(), track_uncertainty);
        assert_eq!(detection.get_position_uncertainty(), detection_uncertainty);
    }

    #[test]
    fn test_prediction_distance_uses_detection_center_simple_blob() {
        check_prediction_distance_uses_detection_center(SimpleBlob::new_with_dt);
    }

    #[test]
    fn test_prediction_distance_uses_detection_center_bbox() {
        check_prediction_distance_uses_detection_center(BlobBBox::new_with_dt);
    }

    fn check_prediction_matching_rejects_distant_detection<B: Blob>(new_blob: fn(Rect, f32) -> B) {
        for offset in [0.0, 1024.0, -1024.0] {
            let mut tracker = super::SimpleTracker::default();
            let mut frame = vec![new_blob(Rect::new(offset, offset, 10.0, 10.0), 0.125)];
            let old_id = frame[0].get_id();
            tracker.match_objects(&mut frame).unwrap();

            let mut frame = vec![new_blob(
                Rect::new(offset + 1000.0, offset + 1000.0, 10.0, 10.0),
                0.125,
            )];
            let new_id = frame[0].get_id();
            tracker.match_objects(&mut frame).unwrap();

            assert_eq!(frame[0].get_id(), new_id, "scene offset: {offset}");
            assert_eq!(tracker.objects.len(), 2);
            assert_eq!(tracker.objects[&old_id].get_no_match_times(), 1);
            assert_eq!(tracker.objects[&new_id].get_no_match_times(), 0);
        }
    }

    #[test]
    fn test_prediction_matching_rejects_distant_detection_simple_blob() {
        check_prediction_matching_rejects_distant_detection(SimpleBlob::new_with_dt);
    }

    #[test]
    fn test_prediction_matching_rejects_distant_detection_bbox() {
        check_prediction_matching_rejects_distant_detection(BlobBBox::new_with_dt);
    }

    fn check_prediction_matching_preserves_moving_track<B: Blob>(new_blob: fn(Rect, f32) -> B) {
        let mut tracker = super::SimpleTracker::default();
        let mut frame = vec![new_blob(Rect::new(100.0, 200.0, 10.0, 10.0), 0.125)];
        let id = frame[0].get_id();
        tracker.match_objects(&mut frame).unwrap();

        for step in 1..=8 {
            let mut frame = vec![new_blob(
                Rect::new(100.0 + step as f32 * 5.0, 200.0, 10.0, 10.0),
                0.125,
            )];
            tracker.match_objects(&mut frame).unwrap();
            assert_eq!(frame[0].get_id(), id);
        }

        // After a longer interval, only the prediction is close enough to match.
        let dt = 2.0;
        let mut expected = tracker.objects[&id].clone();
        expected.set_dt(dt);
        let (x, y) = expected.get_predicted_position_readonly();
        let detection = new_blob(Rect::new(x - 5.0, y - 5.0, 10.0, 10.0), dt);
        assert!(detection.distance_to(&expected) > 30.0);
        expected.predict_next_position();
        expected.update(&detection).unwrap();

        let mut frame = vec![detection];
        tracker.match_objects(&mut frame).unwrap();

        assert_eq!(frame[0].get_id(), id);
        assert_eq!(tracker.objects.len(), 1);
        let track = &tracker.objects[&id];
        assert_eq!(track.get_no_match_times(), 0);
        assert_eq!(track.get_lost_seconds(), 0.0);
        assert_eq!(
            track.get_predicted_position_readonly(),
            expected.get_predicted_position_readonly()
        );
        assert_eq!(
            track.get_position_uncertainty(),
            expected.get_position_uncertainty()
        );
    }

    #[test]
    fn test_prediction_matching_preserves_moving_track_simple_blob() {
        check_prediction_matching_preserves_moving_track(SimpleBlob::new_with_dt);
    }

    #[test]
    fn test_prediction_matching_preserves_moving_track_bbox() {
        check_prediction_matching_preserves_moving_track(BlobBBox::new_with_dt);
    }

    fn check_detection_dt_before_prediction<B: Blob>(new_blob: fn(Rect, f32) -> B) {
        let mut frame = vec![
            new_blob(Rect::new(100.0, 200.0, 30.0, 40.0), 0.125),
            new_blob(Rect::new(1000.0, 200.0, 30.0, 40.0), 0.125),
        ];
        let matched_id = frame[0].get_id();
        let missed_id = frame[1].get_id();
        let mut tracker = super::SimpleTracker::default();
        tracker.match_objects(&mut frame).unwrap();

        let dt = 0.5;
        let detection = new_blob(Rect::new(104.0, 200.0, 30.0, 40.0), dt);
        let mut expected_matched = tracker.objects[&matched_id].clone();
        expected_matched.set_dt(dt);
        expected_matched.predict_next_position();
        expected_matched.update(&detection).unwrap();
        let mut expected_missed = tracker.objects[&missed_id].clone();
        expected_missed.set_dt(dt);
        expected_missed.predict_next_position();

        let mut frame = vec![detection];
        tracker.match_objects(&mut frame).unwrap();

        assert_eq!(frame[0].get_id(), matched_id);
        assert_eq!(tracker.objects.len(), 2);
        for (id, expected) in [(matched_id, expected_matched), (missed_id, expected_missed)] {
            let track = &tracker.objects[&id];
            assert_eq!(track.get_dt(), dt);
            assert_eq!(
                track.get_predicted_position_readonly(),
                expected.get_predicted_position_readonly()
            );
            assert_eq!(
                track.get_position_uncertainty(),
                expected.get_position_uncertainty()
            );
        }
    }

    #[test]
    fn test_detection_dt_before_prediction() {
        check_detection_dt_before_prediction(SimpleBlob::new_with_dt);
        check_detection_dt_before_prediction(BlobBBox::new_with_dt);
    }

    fn check_detection_dt_controls_lost_time<B: Blob>(new_blob: fn(Rect, f32) -> B) {
        let bbox = Rect::new(100.0, 200.0, 30.0, 40.0);
        let mut frame = vec![
            new_blob(bbox.clone(), 0.125),
            new_blob(Rect::new(1000.0, 200.0, 30.0, 40.0), 0.125),
        ];
        let matched_id = frame[0].get_id();
        let missed_id = frame[1].get_id();
        let mut tracker = super::SimpleTracker::default();
        tracker.set_max_lost_seconds(0.75);
        tracker.match_objects(&mut frame).unwrap();

        for (index, (dt, lost_seconds)) in [(0.5, 0.5), (0.25, 0.75)].into_iter().enumerate() {
            let mut frame = vec![new_blob(bbox.clone(), dt)];
            tracker.match_objects(&mut frame).unwrap();
            assert_eq!(frame[0].get_id(), matched_id);
            let matched = &tracker.objects[&matched_id];
            assert_eq!(matched.get_dt(), dt);
            assert_eq!(matched.get_lost_seconds(), 0.0);
            let missed = &tracker.objects[&missed_id];
            assert_eq!(missed.get_dt(), dt);
            assert_eq!(missed.get_no_match_times(), index + 1);
            assert_eq!(missed.get_lost_seconds(), lost_seconds);
        }

        let mut frame = vec![new_blob(bbox, 0.125)];
        tracker.match_objects(&mut frame).unwrap();
        assert_eq!(frame[0].get_id(), matched_id);
        assert_eq!(tracker.objects.len(), 1);
        assert!(!tracker.objects.contains_key(&missed_id));

        // Empty frames keep the last interval unless the caller supplies a new one.
        tracker.match_objects(&mut vec![]).unwrap();
        let track = &tracker.objects[&matched_id];
        assert_eq!(track.get_dt(), 0.125);
        assert_eq!(track.get_no_match_times(), 1);
        assert_eq!(track.get_lost_seconds(), 0.125);

        tracker.set_dt(0.25);
        tracker.match_objects(&mut vec![]).unwrap();
        let track = &tracker.objects[&matched_id];
        assert_eq!(track.get_dt(), 0.25);
        assert_eq!(track.get_no_match_times(), 2);
        assert_eq!(track.get_lost_seconds(), 0.375);
    }

    #[test]
    fn test_detection_dt_controls_lost_time() {
        check_detection_dt_controls_lost_time(SimpleBlob::new_with_dt);
        check_detection_dt_controls_lost_time(BlobBBox::new_with_dt);
    }

    fn check_missed_frame_counters<B: Blob>(new_blob: fn(Rect, f32) -> B) {
        let bbox = Rect::new(100.0, 200.0, 30.0, 40.0);
        let dt = 0.125;
        let mut tracker = super::SimpleTracker::default();
        let mut frame = vec![new_blob(bbox.clone(), dt)];
        let id = frame[0].get_id();

        tracker.match_objects(&mut frame).unwrap();
        let track = tracker.objects.get(&id).unwrap();
        assert_eq!(track.get_no_match_times(), 0);
        assert_eq!(track.get_lost_seconds(), 0.0);

        for _ in 0..3 {
            let mut frame = vec![new_blob(bbox.clone(), dt)];
            tracker.match_objects(&mut frame).unwrap();
            assert_eq!(frame[0].get_id(), id);
            assert_eq!(tracker.objects.len(), 1);
            let track = tracker.objects.get(&id).unwrap();
            assert_eq!(track.get_no_match_times(), 0);
            assert_eq!(track.get_lost_seconds(), 0.0);
        }

        for missed_frames in 1..=2 {
            tracker.match_objects(&mut vec![]).unwrap();
            let track = tracker.objects.get(&id).unwrap();
            assert_eq!(track.get_no_match_times(), missed_frames);
            assert_eq!(track.get_lost_seconds(), missed_frames as f32 * dt);
        }

        let mut frame = vec![new_blob(bbox, dt)];
        tracker.match_objects(&mut frame).unwrap();
        assert_eq!(frame[0].get_id(), id);
        assert_eq!(tracker.objects.len(), 1);
        let track = tracker.objects.get(&id).unwrap();
        assert_eq!(track.get_no_match_times(), 0);
        assert_eq!(track.get_lost_seconds(), 0.0);

        tracker.match_objects(&mut vec![]).unwrap();
        let track = tracker.objects.get(&id).unwrap();
        assert_eq!(track.get_no_match_times(), 1);
        assert_eq!(track.get_lost_seconds(), dt);
    }

    #[test]
    fn test_missed_frame_counters() {
        check_missed_frame_counters(SimpleBlob::new_with_dt);
        check_missed_frame_counters(BlobBBox::new_with_dt);
    }

    fn check_only_unmatched_tracks_count_as_missed<B: Blob>(new_blob: fn(Rect, f32) -> B) {
        let dt = 0.125;
        let matched_bbox = Rect::new(100.0, 200.0, 30.0, 40.0);
        let mut frame = vec![
            new_blob(matched_bbox.clone(), dt),
            new_blob(Rect::new(1000.0, 200.0, 30.0, 40.0), dt),
        ];
        let matched_id = frame[0].get_id();
        let missed_id = frame[1].get_id();
        let mut tracker = super::SimpleTracker::default();
        tracker.match_objects(&mut frame).unwrap();

        let mut frame = vec![
            new_blob(matched_bbox, dt),
            new_blob(Rect::new(2000.0, 200.0, 30.0, 40.0), dt),
        ];
        let new_id = frame[1].get_id();
        tracker.match_objects(&mut frame).unwrap();

        assert_eq!(frame[0].get_id(), matched_id);
        assert_eq!(frame[1].get_id(), new_id);
        assert_eq!(tracker.objects.len(), 3);
        for id in [matched_id, new_id] {
            let track = tracker.objects.get(&id).unwrap();
            assert_eq!(track.get_no_match_times(), 0);
            assert_eq!(track.get_lost_seconds(), 0.0);
        }
        let track = tracker.objects.get(&missed_id).unwrap();
        assert_eq!(track.get_no_match_times(), 1);
        assert_eq!(track.get_lost_seconds(), dt);
    }

    #[test]
    fn test_only_unmatched_tracks_count_as_missed() {
        check_only_unmatched_tracks_count_as_missed(SimpleBlob::new_with_dt);
        check_only_unmatched_tracks_count_as_missed(BlobBBox::new_with_dt);
    }

    fn check_detected_track_survives_limit<B: Blob>(
        mut tracker: super::SimpleTracker<B>,
        new_blob: fn(Rect, f32) -> B,
    ) {
        let bbox = Rect::new(100.0, 200.0, 30.0, 40.0);
        let dt = 0.125;
        let mut frame = vec![new_blob(bbox.clone(), dt)];
        let id = frame[0].get_id();

        tracker.match_objects(&mut frame).unwrap();
        assert!(tracker.objects.contains_key(&id));

        for _ in 0..3 {
            let mut frame = vec![new_blob(bbox.clone(), dt)];
            tracker.match_objects(&mut frame).unwrap();
            assert_eq!(frame[0].get_id(), id);
            assert_eq!(tracker.objects.len(), 1);
            assert!(tracker.objects.contains_key(&id));
        }

        tracker.match_objects(&mut vec![]).unwrap();
        assert!(tracker.objects.is_empty());
    }

    #[test]
    fn test_detected_track_survives_zero_frame_limit() {
        check_detected_track_survives_limit(
            super::SimpleTracker::new(0, 30.0),
            SimpleBlob::new_with_dt,
        );
        check_detected_track_survives_limit(
            super::SimpleTracker::new(0, 30.0),
            BlobBBox::new_with_dt,
        );
    }

    #[test]
    fn test_detected_track_survives_short_time_limit() {
        let mut simple_tracker = super::SimpleTracker::default();
        simple_tracker.set_max_lost_seconds(0.0625);
        check_detected_track_survives_limit(simple_tracker, SimpleBlob::new_with_dt);

        let mut bbox_tracker = super::SimpleTracker::default();
        bbox_tracker.set_max_lost_seconds(0.0625);
        check_detected_track_survives_limit(bbox_tracker, BlobBBox::new_with_dt);
    }

    #[test]
    fn test_match_objects_spread() {
        let bboxes_iterations = get_spread_data();
        let mut mot: super::SimpleTracker<SimpleBlob> = super::SimpleTracker::new(5, 15.0);
        let dt = 1.0 / 25.00; // emulate 25 fps

        for iteration in bboxes_iterations {
            let mut blobs: Vec<SimpleBlob> = iteration
                .into_iter()
                .map(|bbox| SimpleBlob::new_with_dt(bbox, dt))
                .collect();
            match mot.match_objects(&mut blobs) {
                Ok(_) => {}
                Err(err) => {
                    println!("{:?}", err);
                }
            };
        }

        assert_eq!(mot.objects.len(), 4);

        // println!("id;track");
        // for object in &mot.objects {
        //     print!("{};", object.0);
        //     let track = object.1.get_track();
        //     for (idx, pt) in track.iter().enumerate() {
        //         if idx == track.len() - 1 {
        //             print!("{},{}", pt.x, pt.y);
        //         } else {
        //             print!("{},{}|", pt.x, pt.y);
        //         }
        //     }
        //     println!();
        // }
    }

    #[test]
    fn test_match_objects_spread_bbox() {
        use std::collections::HashMap;
        use uuid::Uuid;

        let bboxes_iterations = get_spread_data();
        let mut mot: super::SimpleTracker<BlobBBox> = super::SimpleTracker::new(5, 15.0);
        let dt = 1.0 / 25.00; // emulate 25 fps

        // Collect bbox history during iterations
        let mut bbox_history: HashMap<Uuid, Vec<(f32, f32, f32, f32)>> = HashMap::new();

        for iteration in bboxes_iterations {
            let mut blobs: Vec<BlobBBox> = iteration
                .into_iter()
                .map(|bbox| BlobBBox::new_with_dt(bbox, dt))
                .collect();
            match mot.match_objects(&mut blobs) {
                Ok(_) => {}
                Err(err) => {
                    println!("{:?}", err);
                }
            };
            // Collect current bbox for each tracked object
            for (id, obj) in &mot.objects {
                let bbox = obj.get_bbox();
                let cx = bbox.x + bbox.width / 2.0;
                let cy = bbox.y + bbox.height / 2.0;
                bbox_history.entry(*id).or_default().push((cx, cy, bbox.width, bbox.height));
            }
        }

        assert_eq!(mot.objects.len(), 4);

        // Output format: id;cx,cy,w,h|cx,cy,w,h|...
        // println!("id;track");
        // for (id, history) in &bbox_history {
        //     print!("{};", id);
        //     for (idx, (cx, cy, w, h)) in history.iter().enumerate() {
        //         if idx == history.len() - 1 {
        //             print!("{},{},{},{}", cx, cy, w, h);
        //         } else {
        //             print!("{},{},{},{}|", cx, cy, w, h);
        //         }
        //     }
        //     println!();
        // }
    }

    #[test]
    fn test_match_objects_naive() {
        let (bboxes_one, bboxes_two, bboxes_three) = get_naive_data();
        let mut mot: super::SimpleTracker<SimpleBlob> = super::SimpleTracker::new(5, 15.0);
        let dt = 1.0 / 25.00; // emulate 25 fps

        for (bbox_one, bbox_two, bbox_three) in
            itertools::izip!(bboxes_one, bboxes_two, bboxes_three)
        {
            let blob_one = SimpleBlob::new_with_dt(bbox_to_rect(&bbox_one), dt);
            let blob_two = SimpleBlob::new_with_dt(bbox_to_rect(&bbox_two), dt);
            let blob_three = SimpleBlob::new_with_dt(bbox_to_rect(&bbox_three), dt);

            let mut blobs = vec![blob_one, blob_two, blob_three];
            match mot.match_objects(&mut blobs) {
                Ok(_) => {}
                Err(err) => {
                    println!("{:?}", err);
                }
            };
        }

        assert_eq!(mot.objects.len(), 3);

        // println!("id;track");
        // for object in &mot.objects {
        //     print!("{};", object.0);
        //     let track = object.1.get_track();
        //     for (idx, pt) in track.iter().enumerate() {
        //         if idx == track.len() - 1 {
        //             print!("{},{}", pt.x, pt.y);
        //         } else {
        //             print!("{},{}|", pt.x, pt.y);
        //         }
        //     }
        //     println!();
        // }
    }

    #[test]
    fn test_match_objects_naive_bbox() {
        use std::collections::HashMap;
        use uuid::Uuid;

        let (bboxes_one, bboxes_two, bboxes_three) = get_naive_data();
        let mut mot: super::SimpleTracker<BlobBBox> = super::SimpleTracker::new(5, 15.0);
        let dt = 1.0 / 25.00; // emulate 25 fps

        // Collect bbox history during iterations
        let mut bbox_history: HashMap<Uuid, Vec<(f32, f32, f32, f32)>> = HashMap::new();

        for (bbox_one, bbox_two, bbox_three) in
            itertools::izip!(bboxes_one, bboxes_two, bboxes_three)
        {
            let blob_one = BlobBBox::new_with_dt(bbox_to_rect(&bbox_one), dt);
            let blob_two = BlobBBox::new_with_dt(bbox_to_rect(&bbox_two), dt);
            let blob_three = BlobBBox::new_with_dt(bbox_to_rect(&bbox_three), dt);

            let mut blobs = vec![blob_one, blob_two, blob_three];
            match mot.match_objects(&mut blobs) {
                Ok(_) => {}
                Err(err) => {
                    println!("{:?}", err);
                }
            };
            // Collect current bbox for each tracked object
            for (id, obj) in &mot.objects {
                let bbox = obj.get_bbox();
                let cx = bbox.x + bbox.width / 2.0;
                let cy = bbox.y + bbox.height / 2.0;
                bbox_history.entry(*id).or_default().push((cx, cy, bbox.width, bbox.height));
            }
        }

        assert_eq!(mot.objects.len(), 3);

        // Output format: id;cx,cy,w,h|cx,cy,w,h|...
        // println!("id;track");
        // for (id, history) in &bbox_history {
        //     print!("{};", id);
        //     for (idx, (cx, cy, w, h)) in history.iter().enumerate() {
        //         if idx == history.len() - 1 {
        //             print!("{},{},{},{}", cx, cy, w, h);
        //         } else {
        //             print!("{},{},{},{}|", cx, cy, w, h);
        //         }
        //     }
        //     println!();
        // }
    }

    /// With a time limit the track must go when its unmatched time exceeds the
    /// limit, no matter how generous the frame limit is
    #[test]
    fn test_expiry_by_seconds_simple() {
        let dt = 0.5;
        let mut tracker = crate::mot::SimpleTracker::<crate::mot::SimpleBlob>::new(1000, 30.0);
        tracker.set_max_lost_seconds(1.0);
        assert_eq!(tracker.get_max_lost_seconds(), Some(1.0));
        let mut frame = vec![crate::mot::SimpleBlob::new_with_dt(
            crate::utils::Rect::new(10.0, 10.0, 20.0, 20.0),
            dt,
        )];
        tracker.match_objects(&mut frame).unwrap();
        assert_eq!(tracker.objects.len(), 1);
        // The object vanishes. Each empty frame adds dt to the lost time; the
        // track must survive exactly as long as that stays within the limit
        let mut empty_frames = 0;
        while !tracker.objects.is_empty() {
            let lost = tracker.objects.values().next().unwrap().get_lost_seconds();
            assert!(lost <= 1.0, "track kept while lost for {lost} s > 1 s");
            let mut empty = vec![];
            tracker.match_objects(&mut empty).unwrap();
            empty_frames += 1;
            assert!(empty_frames <= 3, "track not expired by time after {empty_frames} empty frames");
        }
        assert!(empty_frames >= 2, "track expired too early, after {empty_frames} empty frames");
    }

    /// A frame without detections carries no dt, so the tracker must be told the
    /// real interval explicitly for the lost time to be counted right
    #[test]
    fn test_set_dt_on_empty_frames_simple() {
        let nominal_dt = 0.1;
        let mut with_set_dt = crate::mot::SimpleTracker::<crate::mot::SimpleBlob>::new(1000, 30.0);
        with_set_dt.set_max_lost_seconds(1.0);
        let mut without_set_dt = crate::mot::SimpleTracker::<crate::mot::SimpleBlob>::new(1000, 30.0);
        without_set_dt.set_max_lost_seconds(1.0);
        for tracker in [&mut with_set_dt, &mut without_set_dt] {
            let mut frame = vec![crate::mot::SimpleBlob::new_with_dt(
                crate::utils::Rect::new(10.0, 10.0, 20.0, 20.0),
                nominal_dt,
            )];
            tracker.match_objects(&mut frame).unwrap();
            assert_eq!(tracker.objects.len(), 1);
        }
        // Frames now arrive every 0.5 s instead of 0.1 s and the object is gone
        for _ in 0..4 {
            let mut empty = vec![];
            let tracker = &mut with_set_dt;
            tracker.set_dt(0.5);
            tracker.match_objects(&mut empty).unwrap();
            let mut empty = vec![];
            let tracker = &mut without_set_dt;
            tracker.match_objects(&mut empty).unwrap();
        }
        assert!(with_set_dt.objects.is_empty(), "told the real interval: 4 × 0.5 s > 1 s, must be gone");
        assert_eq!(without_set_dt.objects.len(), 1, "still on the nominal interval: 4 × 0.1 s < 1 s, must be kept");
    }

    /// Without a time limit the frame rule is untouched
    #[test]
    fn test_expiry_by_frames_unchanged_simple() {
        let mut tracker = crate::mot::SimpleTracker::<crate::mot::SimpleBlob>::new(1000, 30.0);
        assert_eq!(tracker.get_max_lost_seconds(), None);
        let mut frame = vec![crate::mot::SimpleBlob::new_with_dt(
            crate::utils::Rect::new(10.0, 10.0, 20.0, 20.0),
            100.0,
        )];
        tracker.match_objects(&mut frame).unwrap();
        // Huge dt, but the frame rule does not care: two empty frames are within 1000
        for _ in 0..2 {
            let mut empty = vec![];
            tracker.match_objects(&mut empty).unwrap();
        }
        assert_eq!(tracker.objects.len(), 1);
        // Non-positive limits are ignored, the frame rule stays in effect
        tracker.set_max_lost_seconds(0.0);
        assert_eq!(tracker.get_max_lost_seconds(), None);
    }
}
