use crate::mot::blob::Blob;
use crate::mot::TrackerError;
use crate::utils::{iou, Rect, Point, euclidean_distance};
use pathfinding::{matrix::Matrix, prelude::kuhn_munkres_min};
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

const SCALE_FACTOR: f32 = 1_000_000.0;

/// Algorithm type for matching detections to tracks
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MatchingAlgorithm {
    /// Use the Hungarian algorithm (Kuhn-Munkres) for optimal assignment
    Hungarian,
    /// Use a greedy algorithm for faster but potentially suboptimal assignment
    Greedy,
}

/// Straightforward implementation of Multi-object tracker (MOT) called ByteTrack.
pub struct ByteTracker<B: Blob> {
    /// Maximum number of frames an object can be missing before it is removed
    max_disappeared: usize,
    /// When set, tracks expire by unmatched time instead of by `max_disappeared` frames. Default is None
    max_lost_seconds: Option<f32>,
    /// Maximum distance between two objects to be considered the same
    min_iou: f32,
    /// High detection confidence threshold
    high_thresh: f32,
    /// Low detection confidence threshold
    low_thresh: f32,
    /// Algorithm to use for matching
    algorithm: MatchingAlgorithm,
    /// Storage
    pub objects: HashMap<Uuid, B>,
}

impl<B: Blob> ByteTracker<B> {
    /// Creates default instance of ByteTracker
    ///
    /// Basic usage:
    ///
    /// ```
    /// use mot_rs::mot::{ByteTracker, SimpleBlob};
    /// let mut tracker: ByteTracker<SimpleBlob> = ByteTracker::default();
    /// ```
    pub fn default() -> Self {
        ByteTracker {
            max_disappeared: 5,
            max_lost_seconds: None,
            min_iou: 0.3,
            high_thresh: 0.5,
            low_thresh: 0.3,
            algorithm: MatchingAlgorithm::Hungarian,
            objects: HashMap::new(),
        }
    }
    /// Creates news instance of ByteTracker
    ///
    /// Basic usage:
    ///
    /// ```
    /// use mot_rs::mot::{ByteTracker, SimpleBlob, MatchingAlgorithm};
    /// let max_disappeared = 5;
    /// let min_iou = 0.3;
    /// let high_thresh = 0.5;
    /// let low_thresh = 0.3;
    /// let algorithm = MatchingAlgorithm::Hungarian;
    /// let mut tracker: ByteTracker<SimpleBlob> = ByteTracker::new(
    ///     max_disappeared,
    ///     min_iou,
    ///     high_thresh,
    ///     low_thresh,
    ///     algorithm,
    /// );
    /// ```
    pub fn new(
        max_disappeared: usize,
        min_iou: f32,
        high_thresh: f32,
        low_thresh: f32,
        algorithm: MatchingAlgorithm,
    ) -> Self {
        ByteTracker {
            max_disappeared,
            max_lost_seconds: None,
            min_iou,
            high_thresh,
            low_thresh,
            algorithm,
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
    /// Switches track expiry back to a frame count (a track is removed once it reaches `max_disappeared` missed frames)
    pub fn set_max_disappeared(&mut self, max_disappeared: usize) {
        self.max_disappeared = max_disappeared;
        self.max_lost_seconds = None;
    }
    /// Time-based expiry limit, if enabled
    pub fn get_max_lost_seconds(&self) -> Option<f32> {
        self.max_lost_seconds
    }
    /// Frame-based expiry limit; in effect only while `get_max_lost_seconds` is `None`
    pub fn get_max_disappeared(&self) -> usize {
        self.max_disappeared
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
    /// Whether a track is still alive, i.e. not lost for longer than allowed
    fn is_alive(&self, track: &B) -> bool {
        match self.max_lost_seconds {
            Some(seconds) => track.get_lost_seconds() <= seconds,
            None => track.get_no_match_times() < self.max_disappeared,
        }
    }

    /// Matches objects in the current frame with existing tracks.
    ///
    /// # Arguments
    /// * `detections` - A slice of rectangles representing detected objects.
    /// * `confidences` - A slice of confidence scores for the detected objects.
    ///
    pub fn match_objects(
        &mut self,
        detections: &mut Vec<B>,
        confidences: &[f32],
    ) -> Result<(), TrackerError> {
        if detections.len() != confidences.len() {
            return Err(TrackerError::BadSize(
                format!(
                    "Detections and confidences arrays must have the same length. Conf array size: {}. Detections array size: {}",
                     confidences.len(), detections.len()
                )
            ));
        }

        // Adopt the real interval between calls, reported through the cycle time
        // of the incoming detections. See IoUTracker::match_objects for why
        if let Some(first) = detections.first() {
            let dt = first.get_dt();
            for (_, track) in self.objects.iter_mut() {
                track.set_dt(dt);
            }
        }

        // Get active tracks
        let active_track_ids: Vec<Uuid> = self
            .objects
            .iter()
            .filter(|(_, track)| self.is_alive(track))
            .map(|(id, _)| *id)
            .collect();
        let active_track_bboxes: Vec<(Uuid, Rect)> = active_track_ids
            .iter()
            .map(|id| (*id, self.objects.get(id).unwrap().get_predicted_bbox_readonly())) // ← ИЗМЕНЕНИЕ: используем предсказанный bbox
            .collect();

        // Set of matched tracks for stage 1
        let mut matched_tracks: HashSet<Uuid> = HashSet::new();
        // Set of matched detections for stage 1
        let mut matched_detections: HashSet<usize> = HashSet::new();

        // 1. First stage: Match high confidence detections
        let high_detection_indices: Vec<usize> = detections
            .iter()
            .enumerate()
            .filter(|(i, _)| confidences[*i] >= self.high_thresh)
            .map(|(i, _)| i)
            .collect();

        // Associate high confidence detections with tracks
        // Calculate IoU matrix between tracks and high confidence detections
        if !active_track_bboxes.is_empty() && !high_detection_indices.is_empty() {
            // Create IoU matrix: rows = tracks, columns = detections
            let iou_matrix =
                self.create_iou_matrix(&active_track_bboxes, &high_detection_indices, detections);
            // Perform matching
            let matches =
                self.perform_matching(&iou_matrix, &active_track_bboxes, &high_detection_indices);
            // Process matches
            self.process_matches(
                matches,
                &active_track_bboxes,
                &high_detection_indices,
                &iou_matrix,
                detections,
                &mut matched_tracks,
                &mut matched_detections,
            )?;
        }

        // 2. Second stage: Match low confidence detections with remaining tracks
        let unmatched_track_ids: Vec<Uuid> = active_track_ids
            .into_iter()
            .filter(|id| !matched_tracks.contains(id))
            .collect();
        let unmatched_track_bboxes: Vec<(Uuid, Rect)> = unmatched_track_ids
            .iter()
            .map(|id| (*id, self.objects.get(id).unwrap().get_predicted_bbox_readonly()))
            .collect();
        let low_detection_indices: Vec<usize> = detections
            .iter()
            .enumerate()
            .filter(|(i, _)| {
                confidences[*i] < self.high_thresh && confidences[*i] >= self.low_thresh
            })
            .map(|(i, _)| i)
            .collect();

        // Associate remaining tracks with low confidence detections
        // Second association stage
        if !unmatched_track_bboxes.is_empty() && !low_detection_indices.is_empty() {
            // Create IoU matrix
            let iou_matrix =
                self.create_iou_matrix(&unmatched_track_bboxes, &low_detection_indices, detections);
            // Perform matching
            let matches =
                self.perform_matching(&iou_matrix, &unmatched_track_bboxes, &low_detection_indices);
            // Process matches
            self.process_matches(
                matches,
                &unmatched_track_bboxes,
                &low_detection_indices,
                &iou_matrix,
                detections,
                &mut matched_tracks,
                &mut matched_detections,
            )?;
        }

        // 3. Add new tracks for unmatched high confidence detections
        for &idx in &high_detection_indices {
            if !matched_detections.contains(&idx) {
                let mut new_blob = detections[idx].clone();
                new_blob.activate();
                self.objects.insert(new_blob.get_id(), new_blob);
            }
        }

        // 4. Increment no_match_times for unmatched tracks
        for (_, track) in self.objects.iter_mut() {
            if !matched_tracks.contains(&track.get_id()) {
                // Progress to t+1 even if no match
                track.predict_next_position();
                track.inc_no_match();
            }
        }

        // 5. Remove tracks that have disappeared for too long
        let max_disappeared = self.max_disappeared;
        let max_lost_seconds = self.max_lost_seconds;
        self.objects.retain(|_, track| match max_lost_seconds {
            Some(seconds) => track.get_lost_seconds() <= seconds,
            None => track.get_no_match_times() < max_disappeared,
        });

        Ok(())
    }
    /// Returns a vec of active tracks.
    ///
    pub fn get_active_tracks(&self) -> Vec<&B> {
        self.objects
            .iter()
            .filter(|(_, track)| self.is_alive(track))
            .map(|(_, track)| track)
            .collect()
    }
    /// Helper function to create IoU matrix
    fn create_iou_matrix(
        &self,
        track_bboxes: &[(Uuid, Rect)],
        detection_indices: &[usize],
        detections: &[B],
    ) -> Vec<Vec<f32>> {
        // Pure IoU approach (for retrospective)
        // let mut iou_matrix: Vec<Vec<f32>> = Vec::with_capacity(track_bboxes.len());
        // for (_, track_bbox) in track_bboxes {
        //     let mut row = Vec::with_capacity(detection_indices.len());
        //     for &det_idx in detection_indices {
        //         let det_rect = detections[det_idx].get_bbox();
        //         let iou_val = iou(track_bbox, &det_rect);
        //         row.push(iou_val);
        //     }
        //     iou_matrix.push(row);
        // }
        // iou_matrix
        let mut iou_matrix: Vec<Vec<f32>> = Vec::with_capacity(track_bboxes.len());
        for (_, track_bbox) in track_bboxes {
            let mut row = Vec::with_capacity(detection_indices.len());
            for &det_idx in detection_indices {
                let det_rect = detections[det_idx].get_bbox();
                let iou_val = iou(track_bbox, &det_rect);
                // Hybrid IoU + Distance matching (for better recovery when IoU is zero)
                let combined_score = if iou_val > 0.05 {
                    // Use IoU when it's reasonably high
                    iou_val
                } else {
                    // Combine IoU and distance (favor IoU when available, fallback to distance)
                    let track_center = Point::new(
                        track_bbox.x + track_bbox.width / 2.0,
                        track_bbox.y + track_bbox.height / 2.0
                    );
                    let distance = euclidean_distance(&track_center, &detections[det_idx].get_center());
                    let distance_score = 1.0 / (1.0 + distance * 0.01);
                    // Lower weight for pure distance matching
                    distance_score * 0.3
                };
                
                row.push(combined_score);
            }
            iou_matrix.push(row);
        }
        iou_matrix
    }
    /// Helper function to perform matching using Hungarian or Greedy algorithm
    fn perform_matching(
        &self,
        iou_matrix: &[Vec<f32>],
        track_bboxes: &[(Uuid, Rect)],
        detection_indices: &[usize],
    ) -> Vec<(usize, usize)> {
        match self.algorithm {
            MatchingAlgorithm::Hungarian => {
                let num_tracks = track_bboxes.len();
                let num_detections = detection_indices.len();
                // Ensure we have at least as many columns as rows for Hungarian
                let padded_cols = num_tracks.max(num_detections);
                // Create padded cost matrix
                let cost_data: Vec<i32> = (0..num_tracks)
                    .flat_map(|i| {
                        (0..padded_cols)
                            .map(|j| {
                                if j < num_detections {
                                    // Real IoU value
                                    ((1.0 - iou_matrix[i][j]) * SCALE_FACTOR) as i32
                                } else {
                                    // Dummy detection - very high cost (low IoU)
                                    (SCALE_FACTOR) as i32 // Cost of 1.0 (IoU of 0.0)
                                }
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect();
                let cost_matrix = match Matrix::from_vec(num_tracks, padded_cols, cost_data) {
                    Ok(matrix) => matrix,
                    Err(_) => return Vec::new(), // Return empty matches on error
                };
                let (_, assignments) = kuhn_munkres_min(&cost_matrix);
                // Filter out dummy assignments
                assignments
                    .iter()
                    .enumerate()
                    .filter(|&(_, det_idx)| *det_idx < num_detections)
                    .map(|(track_idx, &det_idx)| (track_idx, det_idx))
                    .collect()
            }
            MatchingAlgorithm::Greedy => {
                self.perform_greedy_matching(iou_matrix, track_bboxes, detection_indices)
            }
        }
    }
    /// Helper function for greedy matching
    fn perform_greedy_matching(
        &self,
        iou_matrix: &[Vec<f32>],
        track_bboxes: &[(Uuid, Rect)],
        detection_indices: &[usize],
    ) -> Vec<(usize, usize)> {
        let mut matches = Vec::new();
        let mut matched_dets = HashSet::new();
        for i in 0..track_bboxes.len() {
            let mut best_iou = self.min_iou;
            let mut best_det_idx = None;
            for j in 0..detection_indices.len() {
                // Skip already matched detections
                if matched_dets.contains(&j) {
                    continue;
                }
                let iou_val = iou_matrix[i][j];
                if iou_val > best_iou {
                    best_iou = iou_val;
                    best_det_idx = Some(j);
                }
            }
            // If found a match
            if let Some(det_idx) = best_det_idx {
                matches.push((i, det_idx));
                matched_dets.insert(det_idx);
            }
        }
        matches
    }
    // Helper function to process matches
    fn process_matches(
        &mut self,
        matches: Vec<(usize, usize)>,
        track_bboxes: &[(Uuid, Rect)],
        detection_indices: &[usize],
        iou_matrix: &[Vec<f32>],
        detections_array: &mut Vec<B>,
        matched_tracks: &mut HashSet<Uuid>,
        matched_detections: &mut HashSet<usize>,
    ) -> Result<(), TrackerError> {
        for (track_idx, det_idx) in matches {
            let iou_val = iou_matrix[track_idx][det_idx];
            // Only consider assignments with IoU above threshold
            if iou_val >= self.min_iou {
                let track_id = track_bboxes[track_idx].0;
                let detection_idx = detection_indices[det_idx];
                // Update track with matched detection
                if let Some(track) = self.objects.get_mut(&track_id) {
                    // Correct order: predict first, then update
                    track.predict_next_position(); // Predict next position t+1
                    track.update(&detections_array[detection_idx])?; // Update with detection at t+1
                    track.reset_no_match();
                    // Hand the track's identity back to the caller's detection, the
                    // way IoUTracker does. Callers read the id straight off the
                    // detections they passed in; without this every detection keeps
                    // the throwaway id it was constructed with, so a perfectly
                    // tracked object looks like a brand new object on every frame
                    detections_array[detection_idx].set_id(track_id);
                    // Mark as matched
                    matched_tracks.insert(track_id);
                    matched_detections.insert(detection_idx);
                }
            }
        }
        Ok(())

    }
}

mod tests {
    use crate::mot::blob::Blob;
    use crate::mot::test_data::{bbox_to_rect, get_naive_data, get_spread_data};
    use crate::mot::{BlobBBox, MatchingAlgorithm, SimpleBlob};

    #[test]
    fn test_match_objects_spread() {
        let bboxes_iterations = get_spread_data();

        // Add confidence scores for each detection - testing both high and low confidence cases
        let confidence_iterations: Vec<Vec<f32>> = vec![
            vec![0.91],                   // Frame 1: 1 high confidence
            vec![0.89],                   // Frame 2
            vec![0.92],                   // Frame 3
            vec![0.88],                   // Frame 4
            vec![0.90],                   // Frame 5
            vec![0.91],                   // Frame 6
            vec![0.87],                   // Frame 7
            vec![0.85],                   // Frame 8
            vec![0.40, 0.89],             // Frame 9: low, high confidence
            vec![0.92],                   // Frame 10
            vec![0.39, 0.91],             // Frame 11: low, high confidence
            vec![0.87, 0.92, 0.38, 0.89], // Frame 12: mixed confidences
            vec![0.41, 0.88, 0.91],       // Frame 13
            vec![0.36, 0.92, 0.89],       // Frame 14
            vec![0.89, 0.41, 0.93],       // Frame 15
            vec![0.87, 0.93, 0.39, 0.88], // Frame 16
            vec![0.90, 0.88, 0.42, 0.86], // Frame 17
            vec![0.91, 0.87, 0.40, 0.89], // Frame 18
            vec![0.89, 0.41, 0.92],       // Frame 19
            vec![0.37, 0.91, 0.89],       // Frame 20
            vec![0.88, 0.42],             // Frame 21
            vec![0.36, 0.93, 0.87],       // Frame 22
            vec![0.91, 0.39, 0.88, 0.90], // Frame 23
            vec![0.43, 0.87, 0.93],       // Frame 24
            vec![0.89, 0.90, 0.38],       // Frame 25
            vec![0.91, 0.37, 0.89],       // Frame 26
            vec![0.88, 0.39, 0.92],       // Frame 27
        ];

        let mut mot: super::ByteTracker<SimpleBlob> =
            super::ByteTracker::new(5, 0.3, 0.5, 0.3, MatchingAlgorithm::Hungarian);
        let dt = 1.0 / 25.00; // emulate 25 fps

        for (i, iteration) in bboxes_iterations.iter().enumerate() {
            let mut blobs: Vec<SimpleBlob> = iteration
                .iter()
                .map(|bbox| SimpleBlob::new_with_dt(bbox.clone(), dt))
                .collect();
            // Get confidences for this frame
            let confidences = &confidence_iterations[i];
            match mot.match_objects(&mut blobs, confidences) {
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

        // Add confidence scores for each detection - testing both high and low confidence cases
        let confidence_iterations: Vec<Vec<f32>> = vec![
            vec![0.91],                   // Frame 1: 1 high confidence
            vec![0.89],                   // Frame 2
            vec![0.92],                   // Frame 3
            vec![0.88],                   // Frame 4
            vec![0.90],                   // Frame 5
            vec![0.91],                   // Frame 6
            vec![0.87],                   // Frame 7
            vec![0.85],                   // Frame 8
            vec![0.40, 0.89],             // Frame 9: low, high confidence
            vec![0.92],                   // Frame 10
            vec![0.39, 0.91],             // Frame 11: low, high confidence
            vec![0.87, 0.92, 0.38, 0.89], // Frame 12: mixed confidences
            vec![0.41, 0.88, 0.91],       // Frame 13
            vec![0.36, 0.92, 0.89],       // Frame 14
            vec![0.89, 0.41, 0.93],       // Frame 15
            vec![0.87, 0.93, 0.39, 0.88], // Frame 16
            vec![0.90, 0.88, 0.42, 0.86], // Frame 17
            vec![0.91, 0.87, 0.40, 0.89], // Frame 18
            vec![0.89, 0.41, 0.92],       // Frame 19
            vec![0.37, 0.91, 0.89],       // Frame 20
            vec![0.88, 0.42],             // Frame 21
            vec![0.36, 0.93, 0.87],       // Frame 22
            vec![0.91, 0.39, 0.88, 0.90], // Frame 23
            vec![0.43, 0.87, 0.93],       // Frame 24
            vec![0.89, 0.90, 0.38],       // Frame 25
            vec![0.91, 0.37, 0.89],       // Frame 26
            vec![0.88, 0.39, 0.92],       // Frame 27
        ];

        let mut mot: super::ByteTracker<BlobBBox> =
            super::ByteTracker::new(5, 0.3, 0.5, 0.3, MatchingAlgorithm::Hungarian);
        let dt = 1.0 / 25.00; // emulate 25 fps

        // Collect bbox history during iterations
        let mut bbox_history: HashMap<Uuid, Vec<(f32, f32, f32, f32)>> = HashMap::new();

        for (i, iteration) in bboxes_iterations.iter().enumerate() {
            let mut blobs: Vec<BlobBBox> = iteration
                .iter()
                .map(|bbox| BlobBBox::new_with_dt(bbox.clone(), dt))
                .collect();
            // Get confidences for this frame
            let confidences = &confidence_iterations[i];
            match mot.match_objects(&mut blobs, confidences) {
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

        let mut mot: super::ByteTracker<SimpleBlob> = super::ByteTracker::new(
            50,  // max_disappeared
            0.1, // min_iou
            0.7, // high_thresh
            0.2, // low_thresh
            MatchingAlgorithm::Hungarian,
        );
        let dt = 1.0 / 25.00; // emulate 25 fps

        for (bbox_one, bbox_two, bbox_three) in
            itertools::izip!(bboxes_one, bboxes_two, bboxes_three)
        {
            let blob_one = SimpleBlob::new_with_dt(bbox_to_rect(&bbox_one), dt);
            let blob_two = SimpleBlob::new_with_dt(bbox_to_rect(&bbox_two), dt);
            let blob_three = SimpleBlob::new_with_dt(bbox_to_rect(&bbox_three), dt);

            let mut blobs = vec![blob_one, blob_two, blob_three];
            // Static confidence scores for testing
            let confidence_scores = vec![0.9, 0.8, 0.7];
            match mot.match_objects(&mut blobs, &confidence_scores) {
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

        let mut mot: super::ByteTracker<BlobBBox> = super::ByteTracker::new(
            50,  // max_disappeared
            0.1, // min_iou
            0.7, // high_thresh
            0.2, // low_thresh
            MatchingAlgorithm::Hungarian,
        );
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
            // Static confidence scores for testing
            let confidence_scores = vec![0.9, 0.8, 0.7];
            match mot.match_objects(&mut blobs, &confidence_scores) {
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
    fn test_expiry_by_seconds_bytetrack() {
        let dt = 0.5;
        let mut tracker = crate::mot::ByteTracker::<crate::mot::SimpleBlob>::new(1000, 0.3, 0.5, 0.3, crate::mot::MatchingAlgorithm::Hungarian);
        tracker.set_max_lost_seconds(1.0);
        assert_eq!(tracker.get_max_lost_seconds(), Some(1.0));
        let mut frame = vec![crate::mot::SimpleBlob::new_with_dt(
            crate::utils::Rect::new(10.0, 10.0, 20.0, 20.0),
            dt,
        )];
        { let conf = vec![0.9; frame.len()]; tracker.match_objects(&mut frame, &conf) }.unwrap();
        assert_eq!(tracker.objects.len(), 1);
        // The object vanishes. Each empty frame adds dt to the lost time; the
        // track must survive exactly as long as that stays within the limit
        let mut empty_frames = 0;
        while !tracker.objects.is_empty() {
            let lost = tracker.objects.values().next().unwrap().get_lost_seconds();
            assert!(lost <= 1.0, "track kept while lost for {lost} s > 1 s");
            let mut empty = vec![];
            { let conf = vec![0.9; empty.len()]; tracker.match_objects(&mut empty, &conf) }.unwrap();
            empty_frames += 1;
            assert!(empty_frames <= 3, "track not expired by time after {empty_frames} empty frames");
        }
        assert!(empty_frames >= 2, "track expired too early, after {empty_frames} empty frames");
    }

    /// A frame without detections carries no dt, so the tracker must be told the
    /// real interval explicitly for the lost time to be counted right
    #[test]
    fn test_set_dt_on_empty_frames_bytetrack() {
        let nominal_dt = 0.1;
        let mut with_set_dt = crate::mot::ByteTracker::<crate::mot::SimpleBlob>::new(1000, 0.3, 0.5, 0.3, crate::mot::MatchingAlgorithm::Hungarian);
        with_set_dt.set_max_lost_seconds(1.0);
        let mut without_set_dt = crate::mot::ByteTracker::<crate::mot::SimpleBlob>::new(1000, 0.3, 0.5, 0.3, crate::mot::MatchingAlgorithm::Hungarian);
        without_set_dt.set_max_lost_seconds(1.0);
        for tracker in [&mut with_set_dt, &mut without_set_dt] {
            let mut frame = vec![crate::mot::SimpleBlob::new_with_dt(
                crate::utils::Rect::new(10.0, 10.0, 20.0, 20.0),
                nominal_dt,
            )];
            { let conf = vec![0.9; frame.len()]; tracker.match_objects(&mut frame, &conf) }.unwrap();
            assert_eq!(tracker.objects.len(), 1);
        }
        // Frames now arrive every 0.5 s instead of 0.1 s and the object is gone
        for _ in 0..4 {
            let mut empty = vec![];
            let tracker = &mut with_set_dt;
            tracker.set_dt(0.5);
            { let conf = vec![0.9; empty.len()]; tracker.match_objects(&mut empty, &conf) }.unwrap();
            let mut empty = vec![];
            let tracker = &mut without_set_dt;
            { let conf = vec![0.9; empty.len()]; tracker.match_objects(&mut empty, &conf) }.unwrap();
        }
        assert!(with_set_dt.objects.is_empty(), "told the real interval: 4 × 0.5 s > 1 s, must be gone");
        assert_eq!(without_set_dt.objects.len(), 1, "still on the nominal interval: 4 × 0.1 s < 1 s, must be kept");
    }

    /// Without a time limit the frame rule is untouched
    #[test]
    fn test_expiry_by_frames_unchanged_bytetrack() {
        let mut tracker = crate::mot::ByteTracker::<crate::mot::SimpleBlob>::new(1000, 0.3, 0.5, 0.3, crate::mot::MatchingAlgorithm::Hungarian);
        assert_eq!(tracker.get_max_lost_seconds(), None);
        let mut frame = vec![crate::mot::SimpleBlob::new_with_dt(
            crate::utils::Rect::new(10.0, 10.0, 20.0, 20.0),
            100.0,
        )];
        { let conf = vec![0.9; frame.len()]; tracker.match_objects(&mut frame, &conf) }.unwrap();
        // Huge dt, but the frame rule does not care: two empty frames are within 1000
        for _ in 0..2 {
            let mut empty = vec![];
            { let conf = vec![0.9; empty.len()]; tracker.match_objects(&mut empty, &conf) }.unwrap();
        }
        assert_eq!(tracker.objects.len(), 1);
        // Non-positive limits are ignored, the frame rule stays in effect
        tracker.set_max_lost_seconds(0.0);
        assert_eq!(tracker.get_max_lost_seconds(), None);
    }
}
