use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

use crate::mot::blob::Blob;
use crate::mot::mot_errors;
use crate::mot::DistanceBlob;
use crate::utils::{
    iou,
    Point,
    euclidean_distance
};
use uuid::Uuid;

/// Naive implementation of Multi-object tracker (MOT) with IoU matching
pub struct IoUTracker<B: Blob> {
    // Max no match (max number of frames when object could not be found again). Default is 75
    max_no_match: usize,
    // When set, tracks expire by unmatched time instead of by `max_no_match` frames. Default is None
    max_lost_seconds: Option<f32>,
    // IoU threshold for matching. Default is 0.3
    iou_threshold: f32,
    // Storage
    pub objects: HashMap<Uuid, B>,
}

impl<B: Blob> IoUTracker<B> {
    /// Creates default instance of IoUTracker
    ///
    /// Basic usage:
    ///
    /// ```
    /// use mot_rs::mot::{IoUTracker, SimpleBlob};
    /// let mut tracker: IoUTracker<SimpleBlob> = IoUTracker::default();
    /// ```
    pub fn default() -> Self {
        IoUTracker {
            max_no_match: 75,
            max_lost_seconds: None,
            iou_threshold: 0.0,
            objects: HashMap::new(),
        }
    }
    /// Creates news instance of IoUTracker
    ///
    /// Basic usage:
    ///
    /// ```
    /// use mot_rs::mot::{IoUTracker, SimpleBlob};
    /// let max_no_match: usize = 100;
    /// let iou_threshold: f32 = 0.3;
    /// let mut tracker: IoUTracker<SimpleBlob> = IoUTracker::new(max_no_match, iou_threshold);
    /// ```
    pub fn new(_max_no_match: usize, _iou_threshold: f32) -> Self {
        IoUTracker {
            max_no_match: _max_no_match,
            max_lost_seconds: None,
            iou_threshold: _iou_threshold,
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

    fn best_available_match(&self, detection: &B, reserved: &HashSet<Uuid>) -> Option<(Uuid, f32)> {
        let mut best_match = None;
        let mut max_score = 0.0;
        for (id, object) in self.objects.iter() {
            if reserved.contains(id) {
                continue;
            }
            let predicted_bbox = object.get_predicted_bbox_readonly();
            let iou_value = iou(&detection.get_bbox(), &predicted_bbox);
            let predicted_center = Point::new(
                predicted_bbox.x + predicted_bbox.width / 2.0,
                predicted_bbox.y + predicted_bbox.height / 2.0,
            );
            let distance = euclidean_distance(&predicted_center, &detection.get_center());
            let distance_score = 1.0 / (1.0 + distance * 0.01);
            // Favor IoU when available, with a lower weight for pure distance matching.
            let combined_score = if iou_value > 0.05 {
                iou_value * 0.8 + distance_score * 0.2
            } else {
                distance_score * 0.5
            };
            if combined_score > max_score {
                max_score = combined_score;
                best_match = Some((*id, combined_score));
            }
        }
        best_match
    }

    // Matches new objects to existing ones
    pub fn match_objects(
        &mut self,
        new_objects: &mut Vec<B>,
    ) -> Result<(), mot_errors::TrackerError> {
        // The caller reports the real time since the previous call through the
        // cycle time of the incoming detections. Existing tracks were built for
        // whatever interval was current when they were created, so rebuild them
        // for this one: predicting a moving object over a nominal 40 ms when
        // 160 ms actually elapsed places the predicted box a whole stride short
        // of the detection, and the match is then lost for no other reason
        if let Some(first) = new_objects.first() {
            let dt = first.get_dt();
            for (_, object) in self.objects.iter_mut() {
                object.set_dt(dt);
            }
        }

        for (_, object) in self.objects.iter_mut() {
            // Make sure that object is marked as deactivated
            object.deactivate();
        }
        let mut blobs_to_register: HashMap<Uuid, B> = HashMap::new();
        // Each existing track can receive only one detection per frame.
        let mut reserved_objects: HashSet<Uuid> = HashSet::new();

        // Add new objects to priority queue
        let mut priority_queue: BinaryHeap<Reverse<DistanceBlob<B>>> = BinaryHeap::new();
        for new_object in new_objects.iter_mut() {
            let Some((max_id, max_iou)) =
                self.best_available_match(new_object, &reserved_objects)
            else {
                blobs_to_register.insert(new_object.get_id(), new_object.clone());
                continue;
            };
            let distance_blob = DistanceBlob {
                distance_metric_value: max_iou,
                min_id: max_id,
                blob: new_object,
            };
            priority_queue.push(Reverse(distance_blob));
        }

        // Process matches with correct temporal order
        while let Some(mut distance_blob) = priority_queue.pop() {
            let max_iou = distance_blob.0.distance_metric_value;
            let min_id = distance_blob.0.min_id;

            if reserved_objects.contains(&min_id) {
                if let Some((id, score)) =
                    self.best_available_match(distance_blob.0.blob, &reserved_objects)
                {
                    distance_blob.0.min_id = id;
                    distance_blob.0.distance_metric_value = score;
                    // Retry at the new score so better pending matches keep priority.
                    priority_queue.push(distance_blob);
                } else {
                    blobs_to_register
                        .insert(distance_blob.0.blob.get_id(), distance_blob.0.blob.clone());
                }
                continue;
            }
            // Filter by min IoU threshold
            if max_iou > self.iou_threshold {
                match self.objects.get_mut(&min_id) {
                    Some(v) => {
                        // Advance time and update in correct order:
                        v.predict_next_position(); // Advance Kalman to t+1
                        v.update(&distance_blob.0.blob)?; // Update with measurement from t+1
                        v.reset_no_match();
                        // Last but not least:
                        // We need to update ID of new object to match existing one (that is why we have &mut in function definition)
                        distance_blob.0.blob.set_id(min_id);
                        reserved_objects.insert(min_id);
                    },
                    None => {
                        return Err(mot_errors::TrackerError::from(mot_errors::NoObjectInTracker{txt: format!("immposible self.objects.get_mut(&min_id). Object ID {:?}. IoU value: {:?}", min_id, max_iou)}));
                    }
                };
            } else {
                // Otherwise register object as a new one
                blobs_to_register
                    .insert(distance_blob.0.blob.get_id(), distance_blob.0.blob.clone());
            }
        }

        // Handle unmatched objects (predict forward for track maintenance)
        for (id, object) in self.objects.iter_mut() {
            if !reserved_objects.contains(id) {
                object.predict_next_position(); // Advance unmatched tracks
                object.inc_no_match();
            }
        }

        // Register new tracks after handling misses so their detection frame
        // neither counts as missed nor advances their Kalman state.
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
impl<B: Blob> fmt::Display for IoUTracker<B> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Maximum no match: {}\n\tIoU threshold: {}",
            self.max_no_match, self.iou_threshold
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::mot::blob::Blob;
    use crate::mot::test_data::{bbox_to_rect, get_naive_data, get_spread_data};
    use crate::mot::{BlobBBox, SimpleBlob};
    use crate::utils::Rect;

    fn check_competing_detections<B: Blob>(
        new_blob: fn(Rect, f32) -> B,
        track_positions: &[f32],
        detections: &[(f32, Option<usize>)],
    ) {
        use itertools::Itertools;
        use std::collections::HashSet;

        for detections in detections.iter().permutations(detections.len()) {
            let dt = 0.04;
            let mut tracker = super::IoUTracker::new(5, 0.3);
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

    fn check_new_track_initialization<B: Blob>(blob: B) {
        let id = blob.get_id();
        let dt = blob.get_dt();
        let mut tracker = super::IoUTracker::default();
        tracker.match_objects(&mut vec![blob.clone()]).unwrap();

        let track = tracker.objects.get(&id).unwrap();
        assert_eq!(track.get_no_match_times(), 0);
        assert_eq!(track.get_lost_seconds(), 0.0);
        assert_eq!(
            track.get_predicted_position_readonly(),
            blob.get_predicted_position_readonly()
        );
        assert_eq!(
            track.get_position_uncertainty(),
            blob.get_position_uncertainty()
        );

        tracker.match_objects(&mut vec![]).unwrap();

        let track = tracker.objects.get(&id).unwrap();
        assert_eq!(track.get_no_match_times(), 1);
        assert_eq!(track.get_lost_seconds(), dt);
        let mut expected = blob;
        expected.predict_next_position();
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
    fn test_new_track_initialization() {
        let bbox = Rect::new(10.0, 20.0, 30.0, 40.0);
        check_new_track_initialization(SimpleBlob::new_with_dt(bbox.clone(), 0.04));
        check_new_track_initialization(BlobBBox::new_with_dt(bbox, 0.04));
    }

    fn check_new_track_with_frame_limit<B: Blob>(blob: B) {
        let id = blob.get_id();
        let mut tracker = super::IoUTracker::default();
        tracker.set_max_no_match(0);

        tracker.match_objects(&mut vec![blob]).unwrap();
        assert!(tracker.objects.contains_key(&id));

        tracker.match_objects(&mut vec![]).unwrap();
        assert!(tracker.objects.is_empty());
    }

    #[test]
    fn test_new_track_survives_frame_limit() {
        let bbox = Rect::new(10.0, 20.0, 30.0, 40.0);
        check_new_track_with_frame_limit(SimpleBlob::new_with_dt(bbox.clone(), 0.04));
        check_new_track_with_frame_limit(BlobBBox::new_with_dt(bbox, 0.04));
    }

    fn check_new_track_with_time_limit<B: Blob>(blob: B) {
        let id = blob.get_id();
        let mut tracker = super::IoUTracker::default();
        tracker.set_max_lost_seconds(blob.get_dt() / 2.0);

        tracker.match_objects(&mut vec![blob]).unwrap();
        assert!(tracker.objects.contains_key(&id));

        tracker.match_objects(&mut vec![]).unwrap();
        assert!(tracker.objects.is_empty());
    }

    #[test]
    fn test_new_track_survives_time_limit() {
        let bbox = Rect::new(10.0, 20.0, 30.0, 40.0);
        check_new_track_with_time_limit(SimpleBlob::new_with_dt(bbox.clone(), 0.04));
        check_new_track_with_time_limit(BlobBBox::new_with_dt(bbox, 0.04));
    }

    #[test]
    fn test_match_objects_spread() {
        let bboxes_iterations = get_spread_data();
        let mut mot: super::IoUTracker<SimpleBlob> = super::IoUTracker::new(5, 0.3);
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
        let mut mot: super::IoUTracker<BlobBBox> = super::IoUTracker::new(5, 0.3);
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
        let mut mot: super::IoUTracker<SimpleBlob> = super::IoUTracker::new(5, 0.3);
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
        let mut mot: super::IoUTracker<BlobBBox> = super::IoUTracker::new(5, 0.3);
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
    fn test_expiry_by_seconds_iou() {
        let dt = 0.5;
        let mut tracker = crate::mot::IoUTracker::<crate::mot::SimpleBlob>::new(1000, 0.3);
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
    fn test_set_dt_on_empty_frames_iou() {
        let nominal_dt = 0.1;
        let mut with_set_dt = crate::mot::IoUTracker::<crate::mot::SimpleBlob>::new(1000, 0.3);
        with_set_dt.set_max_lost_seconds(1.0);
        let mut without_set_dt = crate::mot::IoUTracker::<crate::mot::SimpleBlob>::new(1000, 0.3);
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
    fn test_expiry_by_frames_unchanged_iou() {
        let mut tracker = crate::mot::IoUTracker::<crate::mot::SimpleBlob>::new(1000, 0.3);
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
