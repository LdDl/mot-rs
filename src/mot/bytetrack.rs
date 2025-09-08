use crate::mot::SimpleBlob;
use crate::mot::TrackerError;
use crate::utils::{iou, Rect, Point, euclidean_distance};
use pathfinding::{matrix::Matrix, prelude::kuhn_munkres_min};
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::error::Error;
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
pub struct ByteTracker {
    /// Maximum number of frames an object can be missing before it is removed
    max_disappeared: usize,
    /// Maximum distance between two objects to be considered the same
    min_iou: f32,
    /// High detection confidence threshold
    high_thresh: f32,
    /// Low detection confidence threshold
    low_thresh: f32,
    /// Algorithm to use for matching
    algorithm: MatchingAlgorithm,
    /// Storage
    pub objects: HashMap<Uuid, SimpleBlob>,
}

impl ByteTracker {
    /// Creates default instance of ByteTracker
    ///
    /// Basic usage:
    ///
    /// ```
    /// use mot_rs::mot::ByteTracker;
    /// let mut tracker = ByteTracker::default();
    /// ```
    pub fn default() -> Self {
        ByteTracker {
            max_disappeared: 5,
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
    /// use mot_rs::mot::ByteTracker;
    /// let max_disappeared = 5;
    /// let min_iou = 0.3;
    /// let high_thresh = 0.5;
    /// let low_thresh = 0.3;
    /// let algorithm = MatchingAlgorithm::Hungarian;
    /// let mut tracker = ByteTracker::new(
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
            min_iou,
            high_thresh,
            low_thresh,
            algorithm,
            objects: HashMap::new(),
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
        detections: &mut Vec<SimpleBlob>,
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

        // Predict next positions for all existing tracks via Kalman filter
        // for (_, track) in self.objects.iter_mut() {
        //     track.predict_next_position();
        // }

        // Get active tracks
        let active_track_ids: Vec<Uuid> = self
            .objects
            .iter()
            .filter(|(_, track)| track.get_no_match_times() < self.max_disappeared)
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
            .map(|id| (*id, self.objects.get(id).unwrap().get_bbox()))
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
        self.objects
            .retain(|_, track| track.get_no_match_times() < self.max_disappeared);

        Ok(())
    }
    /// Returns a vec of active tracks.
    ///
    pub fn get_active_tracks(&self) -> Vec<&SimpleBlob> {
        self.objects
            .iter()
            .filter(|(_, track)| track.get_no_match_times() < self.max_disappeared)
            .map(|(_, track)| track)
            .collect()
    }
    /// Helper function to create IoU matrix
    fn create_iou_matrix(
        &self,
        track_bboxes: &[(Uuid, Rect)],
        detection_indices: &[usize],
        detections: &[SimpleBlob],
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
                    .filter(|(_, &det_idx)| det_idx < num_detections)
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
        detections_array: &mut Vec<SimpleBlob>,
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
    use crate::mot::{ByteTracker, MatchingAlgorithm, SimpleBlob};
    use crate::utils::Rect;

    #[test]
    fn test_match_objects_spread() {
        let bboxes_iterations: Vec<Vec<Rect>> = vec![
            // Each nested vector represents set of bounding boxes on a single frame
            vec![Rect::new(378.0, 147.0, 173.0, 243.0)],
            vec![Rect::new(374.0, 147.0, 180.0, 253.0)],
            vec![Rect::new(375.0, 154.0, 178.0, 256.0)],
            vec![Rect::new(376.0, 162.0, 177.0, 267.0)],
            vec![Rect::new(375.0, 166.0, 178.0, 268.0)],
            vec![Rect::new(375.0, 177.0, 186.0, 266.0)],
            vec![Rect::new(370.0, 185.0, 197.0, 273.0)],
            vec![Rect::new(363.0, 209.0, 203.0, 264.0)],
            vec![
                Rect::new(70.0, 14.0, 227.0, 254.0),
                Rect::new(364.0, 214.0, 200.0, 262.0),
            ],
            vec![Rect::new(365.0, 218.0, 205.0, 263.0)],
            vec![
                Rect::new(67.0, 23.0, 236.0, 246.0),
                Rect::new(366.0, 231.0, 209.0, 260.0),
            ],
            vec![
                Rect::new(73.0, 18.0, 227.0, 264.0),
                Rect::new(610.0, 47.0, 324.0, 355.0),
                Rect::new(370.0, 238.0, 199.0, 259.0),
                Rect::new(381.0, -1.0, 103.0, 60.0),
            ],
            vec![
                Rect::new(67.0, 16.0, 229.0, 271.0),
                Rect::new(370.0, 250.0, 195.0, 264.0),
                Rect::new(381.0, -2.0, 106.0, 58.0),
            ],
            vec![
                Rect::new(62.0, 15.0, 233.0, 268.0),
                Rect::new(365.0, 257.0, 205.0, 264.0),
                Rect::new(379.0, -1.0, 109.0, 59.0),
            ],
            vec![
                Rect::new(60.0, 7.0, 234.0, 279.0),
                Rect::new(360.0, 269.0, 212.0, 260.0),
                Rect::new(380.0, -1.0, 109.0, 60.0),
            ],
            vec![
                Rect::new(50.0, 41.0, 251.0, 295.0),
                Rect::new(619.0, 25.0, 308.0, 399.0),
                Rect::new(361.0, 276.0, 215.0, 265.0),
                Rect::new(380.0, -1.0, 110.0, 63.0),
            ],
            vec![
                Rect::new(48.0, 36.0, 242.0, 302.0),
                Rect::new(622.0, 21.0, 299.0, 411.0),
                Rect::new(357.0, 283.0, 222.0, 255.0),
                Rect::new(379.0, 0.0, 113.0, 64.0),
            ],
            vec![
                Rect::new(41.0, 28.0, 245.0, 319.0),
                Rect::new(625.0, 31.0, 308.0, 392.0),
                Rect::new(350.0, 306.0, 239.0, 231.0),
                Rect::new(377.0, 0.0, 116.0, 65.0),
            ],
            vec![
                Rect::new(630.0, 98.0, 294.0, 324.0),
                Rect::new(346.0, 310.0, 250.0, 239.0),
                Rect::new(378.0, 0.0, 112.0, 65.0),
            ],
            vec![
                Rect::new(636.0, 99.0, 290.0, 323.0),
                Rect::new(344.0, 320.0, 254.0, 229.0),
                Rect::new(378.0, 2.0, 114.0, 65.0),
            ],
            vec![
                Rect::new(636.0, 103.0, 295.0, 318.0),
                Rect::new(347.0, 332.0, 251.0, 211.0),
            ],
            vec![
                Rect::new(362.0, 1.0, 147.0, 90.0),
                Rect::new(637.0, 104.0, 292.0, 321.0),
                Rect::new(337.0, 344.0, 272.0, 196.0),
            ],
            vec![
                Rect::new(360.0, -2.0, 152.0, 97.0),
                Rect::new(12.0, 74.0, 237.0, 324.0),
                Rect::new(639.0, 104.0, 293.0, 316.0),
                Rect::new(347.0, 350.0, 258.0, 185.0),
            ],
            vec![
                Rect::new(361.0, -4.0, 149.0, 99.0),
                Rect::new(9.0, 112.0, 251.0, 313.0),
                Rect::new(627.0, 106.0, 314.0, 321.0),
            ],
            vec![
                Rect::new(360.0, -3.0, 151.0, 99.0),
                Rect::new(15.0, 115.0, 231.0, 311.0),
                Rect::new(633.0, 91.0, 297.0, 346.0),
            ],
            vec![
                Rect::new(362.0, -7.0, 148.0, 106.0),
                Rect::new(10.0, 109.0, 241.0, 320.0),
                Rect::new(639.0, 93.0, 294.0, 347.0),
            ],
            vec![
                Rect::new(362.0, -9.0, 146.0, 109.0),
                Rect::new(12.0, 109.0, 233.0, 326.0),
                Rect::new(639.0, 95.0, 288.0, 347.0),
            ],
        ];

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

        let mut mot = ByteTracker::new(5, 0.3, 0.5, 0.3, MatchingAlgorithm::Hungarian);
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
    fn test_match_objects_naive() {
        let bboxes_one: Vec<Vec<i32>> = vec![
            vec![236, -25, 386, 35],
            vec![237, -24, 387, 36],
            vec![238, -22, 388, 38],
            vec![236, -20, 386, 40],
            vec![236, -19, 386, 41],
            vec![237, -18, 387, 42],
            vec![237, -18, 387, 42],
            vec![238, -17, 388, 43],
            vec![237, -14, 387, 46],
            vec![237, -14, 387, 46],
            vec![237, -12, 387, 48],
            vec![237, -12, 387, 48],
            vec![237, -11, 387, 49],
            vec![237, -11, 387, 49],
            vec![237, -10, 387, 50],
            vec![237, -10, 387, 50],
            vec![237, -8, 387, 52],
            vec![237, -8, 387, 52],
            vec![236, -7, 386, 53],
            vec![236, -7, 386, 53],
            vec![236, -6, 386, 54],
            vec![236, -6, 386, 54],
            vec![236, -2, 386, 58],
            vec![235, 0, 385, 60],
            vec![236, 2, 386, 62],
            vec![236, 5, 386, 65],
            vec![236, 9, 386, 69],
            vec![235, 12, 385, 72],
            vec![235, 14, 385, 74],
            vec![233, 16, 383, 76],
            vec![232, 26, 382, 86],
            vec![233, 28, 383, 88],
            vec![233, 40, 383, 100],
            vec![233, 30, 383, 90],
            vec![232, 22, 382, 82],
            vec![232, 34, 382, 94],
            vec![232, 21, 382, 81],
            vec![233, 40, 383, 100],
            vec![232, 40, 382, 100],
            vec![232, 40, 382, 100],
            vec![232, 36, 382, 96],
            vec![232, 53, 382, 113],
            vec![232, 50, 382, 110],
            vec![233, 55, 383, 115],
            vec![232, 50, 382, 110],
            vec![234, 68, 384, 128],
            vec![231, 49, 381, 109],
            vec![232, 68, 382, 128],
            vec![231, 31, 381, 91],
            vec![232, 64, 382, 124],
            vec![233, 71, 383, 131],
            vec![231, 64, 381, 124],
            vec![231, 74, 381, 134],
            vec![231, 64, 381, 124],
            vec![230, 77, 380, 137],
            vec![232, 82, 382, 142],
            vec![232, 78, 382, 138],
            vec![232, 78, 382, 138],
            vec![231, 79, 381, 139],
            vec![231, 79, 381, 139],
            vec![231, 91, 381, 151],
            vec![232, 78, 382, 138],
            vec![232, 78, 382, 138],
            vec![233, 90, 383, 150],
            vec![232, 92, 382, 152],
            vec![232, 92, 382, 152],
            vec![233, 98, 383, 158],
            vec![232, 100, 382, 160],
            vec![231, 92, 381, 152],
            vec![233, 110, 383, 170],
            vec![234, 92, 384, 152],
            vec![234, 92, 384, 152],
            vec![234, 110, 384, 170],
            vec![234, 92, 384, 152],
            vec![233, 104, 383, 164],
            vec![234, 111, 384, 171],
            vec![234, 106, 384, 166],
            vec![234, 106, 384, 166],
            vec![233, 124, 383, 184],
            vec![236, 125, 386, 185],
            vec![236, 125, 386, 185],
            vec![232, 120, 382, 180],
            vec![236, 131, 386, 191],
            vec![232, 132, 382, 192],
            vec![238, 139, 388, 199],
            vec![236, 141, 386, 201],
            vec![232, 151, 382, 211],
            vec![236, 145, 386, 205],
            vec![236, 145, 386, 205],
            vec![231, 133, 381, 193],
            vec![237, 148, 387, 208],
            vec![237, 148, 387, 208],
            vec![237, 148, 387, 208],
            vec![237, 148, 387, 208],
            vec![237, 148, 387, 208],
            vec![237, 148, 387, 208],
            vec![237, 148, 387, 208],
            vec![237, 148, 387, 208],
            vec![237, 148, 387, 208],
            vec![237, 148, 387, 208],
            vec![237, 148, 387, 208],
            vec![237, 148, 387, 208],
            vec![237, 148, 387, 208],
            vec![237, 148, 387, 208],
            vec![237, 148, 387, 208],
            vec![237, 148, 387, 208],
            vec![237, 148, 387, 208],
            vec![237, 148, 387, 208],
            vec![237, 148, 387, 208],
            vec![237, 148, 387, 208],
            vec![237, 148, 387, 208],
            vec![237, 148, 387, 208],
        ];
        let bboxes_two: Vec<Vec<i32>> = vec![
            vec![321, -25, 471, 35],
            vec![322, -24, 472, 36],
            vec![323, -22, 473, 38],
            vec![321, -20, 471, 40],
            vec![321, -19, 471, 41],
            vec![322, -18, 472, 42],
            vec![322, -18, 472, 42],
            vec![323, -17, 473, 43],
            vec![322, -14, 472, 46],
            vec![322, -14, 472, 46],
            vec![322, -12, 472, 48],
            vec![322, -12, 472, 48],
            vec![322, -11, 472, 49],
            vec![322, -11, 472, 49],
            vec![322, -10, 472, 50],
            vec![322, -10, 472, 50],
            vec![322, -8, 472, 52],
            vec![322, -8, 472, 52],
            vec![321, -7, 471, 53],
            vec![321, -7, 471, 53],
            vec![321, -6, 471, 54],
            vec![321, -6, 471, 54],
            vec![321, -2, 471, 58],
            vec![320, 0, 470, 60],
            vec![321, 2, 471, 62],
            vec![321, 5, 471, 65],
            vec![321, 9, 471, 69],
            vec![320, 12, 470, 72],
            vec![320, 14, 470, 74],
            vec![318, 16, 468, 76],
            vec![317, 26, 467, 86],
            vec![318, 28, 468, 88],
            vec![318, 40, 468, 100],
            vec![318, 30, 468, 90],
            vec![317, 22, 467, 82],
            vec![317, 34, 467, 94],
            vec![317, 21, 467, 81],
            vec![318, 40, 468, 100],
            vec![317, 40, 467, 100],
            vec![317, 40, 467, 100],
            vec![317, 36, 467, 96],
            vec![317, 53, 467, 113],
            vec![317, 50, 467, 110],
            vec![318, 55, 468, 115],
            vec![317, 50, 467, 110],
            vec![319, 68, 469, 128],
            vec![316, 49, 466, 109],
            vec![317, 68, 467, 128],
            vec![316, 31, 466, 91],
            vec![317, 64, 467, 124],
            vec![318, 71, 468, 131],
            vec![316, 64, 466, 124],
            vec![316, 74, 466, 134],
            vec![316, 64, 466, 124],
            vec![315, 77, 465, 137],
            vec![317, 82, 467, 142],
            vec![317, 78, 467, 138],
            vec![317, 78, 467, 138],
            vec![316, 79, 466, 139],
            vec![316, 79, 466, 139],
            vec![316, 91, 466, 151],
            vec![317, 78, 467, 138],
            vec![317, 78, 467, 138],
            vec![318, 90, 468, 150],
            vec![317, 92, 467, 152],
            vec![317, 92, 467, 152],
            vec![318, 98, 468, 158],
            vec![317, 100, 467, 160],
            vec![316, 92, 466, 152],
            vec![318, 110, 468, 170],
            vec![319, 92, 469, 152],
            vec![319, 92, 469, 152],
            vec![319, 110, 469, 170],
            vec![319, 92, 469, 152],
            vec![318, 104, 468, 164],
            vec![319, 111, 469, 171],
            vec![319, 106, 469, 166],
            vec![319, 106, 469, 166],
            vec![318, 124, 468, 184],
            vec![321, 125, 471, 185],
            vec![321, 125, 471, 185],
            vec![317, 120, 467, 180],
            vec![321, 131, 471, 191],
            vec![317, 132, 467, 192],
            vec![323, 139, 473, 199],
            vec![321, 141, 471, 201],
            vec![317, 151, 467, 211],
            vec![321, 145, 471, 205],
            vec![321, 145, 471, 205],
            vec![316, 133, 466, 193],
            vec![322, 148, 472, 208],
            vec![322, 148, 472, 208],
            vec![322, 148, 472, 208],
            vec![322, 148, 472, 208],
            vec![322, 148, 472, 208],
            vec![322, 148, 472, 208],
            vec![322, 148, 472, 208],
            vec![322, 148, 472, 208],
            vec![322, 148, 472, 208],
            vec![322, 148, 472, 208],
            vec![322, 148, 472, 208],
            vec![322, 148, 472, 208],
            vec![322, 148, 472, 208],
            vec![322, 148, 472, 208],
            vec![322, 148, 472, 208],
            vec![322, 148, 472, 208],
            vec![322, 148, 472, 208],
            vec![322, 148, 472, 208],
            vec![322, 148, 472, 208],
            vec![322, 148, 472, 208],
            vec![322, 148, 472, 208],
            vec![322, 148, 472, 208],
        ];
        let bboxes_three: Vec<Vec<i32>> = vec![
            vec![151, -25, 301, 35],
            vec![152, -24, 302, 36],
            vec![153, -22, 303, 38],
            vec![151, -20, 301, 40],
            vec![151, -19, 301, 41],
            vec![152, -18, 302, 42],
            vec![152, -18, 302, 42],
            vec![153, -17, 303, 43],
            vec![152, -14, 302, 46],
            vec![152, -14, 302, 46],
            vec![152, -12, 302, 48],
            vec![152, -12, 302, 48],
            vec![152, -11, 302, 49],
            vec![152, -11, 302, 49],
            vec![152, -10, 302, 50],
            vec![152, -10, 302, 50],
            vec![152, -8, 302, 52],
            vec![152, -8, 302, 52],
            vec![151, -7, 301, 53],
            vec![151, -7, 301, 53],
            vec![151, -6, 301, 54],
            vec![151, -6, 301, 54],
            vec![151, -2, 301, 58],
            vec![150, 0, 300, 60],
            vec![151, 2, 301, 62],
            vec![151, 5, 301, 65],
            vec![151, 9, 301, 69],
            vec![150, 12, 300, 72],
            vec![150, 14, 300, 74],
            vec![148, 16, 298, 76],
            vec![147, 26, 297, 86],
            vec![148, 28, 298, 88],
            vec![148, 40, 298, 100],
            vec![148, 30, 298, 90],
            vec![147, 22, 297, 82],
            vec![147, 34, 297, 94],
            vec![147, 21, 297, 81],
            vec![148, 40, 298, 100],
            vec![147, 40, 297, 100],
            vec![147, 40, 297, 100],
            vec![147, 36, 297, 96],
            vec![147, 53, 297, 113],
            vec![147, 50, 297, 110],
            vec![148, 55, 298, 115],
            vec![147, 50, 297, 110],
            vec![149, 68, 299, 128],
            vec![146, 49, 296, 109],
            vec![147, 68, 297, 128],
            vec![146, 31, 296, 91],
            vec![147, 64, 297, 124],
            vec![148, 71, 298, 131],
            vec![146, 64, 296, 124],
            vec![146, 74, 296, 134],
            vec![146, 64, 296, 124],
            vec![145, 77, 295, 137],
            vec![147, 82, 297, 142],
            vec![147, 78, 297, 138],
            vec![147, 78, 297, 138],
            vec![146, 79, 296, 139],
            vec![146, 79, 296, 139],
            vec![146, 91, 296, 151],
            vec![147, 78, 297, 138],
            vec![147, 78, 297, 138],
            vec![148, 90, 298, 150],
            vec![147, 92, 297, 152],
            vec![147, 92, 297, 152],
            vec![148, 98, 298, 158],
            vec![147, 100, 297, 160],
            vec![146, 92, 296, 152],
            vec![148, 110, 298, 170],
            vec![149, 92, 299, 152],
            vec![149, 92, 299, 152],
            vec![149, 110, 299, 170],
            vec![149, 92, 299, 152],
            vec![148, 104, 298, 164],
            vec![149, 111, 299, 171],
            vec![149, 106, 299, 166],
            vec![149, 106, 299, 166],
            vec![148, 124, 298, 184],
            vec![151, 125, 301, 185],
            vec![151, 125, 301, 185],
            vec![147, 120, 297, 180],
            vec![151, 131, 301, 191],
            vec![147, 132, 297, 192],
            vec![153, 139, 303, 199],
            vec![151, 141, 301, 201],
            vec![147, 151, 297, 211],
            vec![151, 145, 301, 205],
            vec![151, 145, 301, 205],
            vec![146, 133, 296, 193],
            vec![152, 148, 302, 208],
            vec![152, 148, 302, 208],
            vec![152, 148, 302, 208],
            vec![152, 148, 302, 208],
            vec![152, 148, 302, 208],
            vec![152, 148, 302, 208],
            vec![152, 148, 302, 208],
            vec![152, 148, 302, 208],
            vec![152, 148, 302, 208],
            vec![152, 148, 302, 208],
            vec![152, 148, 302, 208],
            vec![152, 148, 302, 208],
            vec![152, 148, 302, 208],
            vec![152, 148, 302, 208],
            vec![152, 148, 302, 208],
            vec![152, 148, 302, 208],
            vec![152, 148, 302, 208],
            vec![152, 148, 302, 208],
            vec![152, 148, 302, 208],
            vec![152, 148, 302, 208],
            vec![152, 148, 302, 208],
            vec![152, 148, 302, 208],
        ];

        let mut tracker = ByteTracker::new(
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
            let blob_one = super::SimpleBlob::new_with_dt(
                Rect::new(
                    bbox_one[0] as f32,
                    bbox_one[1] as f32,
                    (bbox_one[2] - bbox_one[0]) as f32,
                    (bbox_one[3] - bbox_one[1]) as f32,
                ),
                dt,
            );
            let blob_two = super::SimpleBlob::new_with_dt(
                Rect::new(
                    bbox_two[0] as f32,
                    bbox_two[1] as f32,
                    (bbox_two[2] - bbox_two[0]) as f32,
                    (bbox_two[3] - bbox_two[1]) as f32,
                ),
                dt,
            );
            let blob_three = super::SimpleBlob::new_with_dt(
                Rect::new(
                    bbox_three[0] as f32,
                    bbox_three[1] as f32,
                    (bbox_three[2] - bbox_three[0]) as f32,
                    (bbox_three[3] - bbox_three[1]) as f32,
                ),
                dt,
            );

            let mut blobs = vec![blob_one, blob_two, blob_three];
            // Static confidence scores for testing
            let confidence_scores = vec![0.9, 0.8, 0.7];
            // for blob in blobs.iter() {
            //     println!("id before: {:?}", blob.get_id());
            // }
            match tracker.match_objects(&mut blobs, &confidence_scores) {
                Ok(_) => {}
                Err(err) => {
                    println!("{:?}", err);
                }
            };
            // for blob in blobs.iter() {
            //     println!("\tid after: {:?}", blob.get_id());
            // }
        }

        assert_eq!(tracker.objects.len(), 3);

        // println!("id;track");
        // for object in &tracker.objects {
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
}
