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
            iou_threshold: _iou_threshold,
            objects: HashMap::new(),
        }
    }
    // Matches new objects to existing ones
    pub fn match_objects(
        &mut self,
        new_objects: &mut Vec<B>,
    ) -> Result<(), mot_errors::TrackerError> {
        for (_, object) in self.objects.iter_mut() {
            // Make sure that object is marked as deactivated
            object.deactivate();
        }
        let mut blobs_to_register: HashMap<Uuid, B> = HashMap::new();

        // Add new objects to priority queue
        let mut priority_queue: BinaryHeap<Reverse<DistanceBlob<B>>> = BinaryHeap::new();
        // Calculate IoU using PREDICTED positions
        for new_object in new_objects.iter_mut() {
            // Find existing blob with min distance to new one
            let mut max_id = Uuid::default();
            let mut max_iou = 0.0;

            // Simple IoU matching (for restospective)
            // for (j, object) in self.objects.iter() {
            //     // let iou_value = iou(&new_object.get_bbox(), &object.get_bbox());
            //     // Use predicted bbox for better matching
            //     let predicted_bbox = object.get_predicted_bbox_readonly();
            //     let iou_value = iou(&new_object.get_bbox(), &predicted_bbox);
            //     if iou_value > max_iou {
            //         max_iou = iou_value;
            //         max_id = *j;
            //     }
            // }

            // Hybrid IoU + Distance matching (for better recovery when IoU is zero)
            for (j, object) in self.objects.iter() {
                let predicted_bbox = object.get_predicted_bbox_readonly();
                let iou_value = iou(&new_object.get_bbox(), &predicted_bbox);
                // Add distance-based fallback
                let predicted_center = Point::new(
                    predicted_bbox.x + predicted_bbox.width / 2.0,
                    predicted_bbox.y + predicted_bbox.height / 2.0
                );
                let distance = euclidean_distance(&predicted_center, &new_object.get_center());
                // Convert to 0-1 similarity
                let distance_score = 1.0 / (1.0 + distance * 0.01);
                // Combine IoU and distance (favor IoU when available, fallback to distance)
                let combined_score = if iou_value > 0.05 { 
                    iou_value * 0.8 + distance_score * 0.2
                } else {
                    // Lower weight for pure distance matching
                    distance_score * 0.5
                };
                if combined_score > max_iou {
                    max_iou = combined_score;
                    max_id = *j;
                }
            }
            let distance_blob = DistanceBlob {
                distance_metric_value: max_iou,
                min_id: max_id,
                blob: new_object,
            };
            priority_queue.push(Reverse(distance_blob));
        }

        // We need to prevent double update of objects
        let mut reserved_objects: HashSet<Uuid> = HashSet::new();

        // Process matches with correct temporal order
        while let Some(distance_blob) = priority_queue.pop() {
            let max_iou = distance_blob.0.distance_metric_value;
            let min_id = distance_blob.0.min_id;

            // Check if object is already reserved
            // Since we are using priority queue with min-heap then we garantee that we will update existing objects with min distance only once.
            // For other objects with the same min_id we can create new objects
            if reserved_objects.contains(&min_id) {
                // Register it immediately and continue
                blobs_to_register
                    .insert(distance_blob.0.blob.get_id(), distance_blob.0.blob.clone());
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

        self.objects.extend(blobs_to_register);

        // Handle unmatched objects (predict forward for track maintenance)
        for (id, object) in self.objects.iter_mut() {
            if !reserved_objects.contains(id) {
                object.predict_next_position(); // Advance unmatched tracks
                object.inc_no_match();
            }
        }

        // Clean up existing data
        self.objects.retain(|_, object| {
            // Remove object if it was not found for a long time
            let delete = object.get_no_match_times() > self.max_no_match;
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

mod tests {
    use crate::mot::blob::Blob;
    use crate::mot::test_data::{bbox_to_rect, get_naive_data, get_spread_data};
    use crate::mot::{BlobBBox, SimpleBlob};

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
        let bboxes_iterations = get_spread_data();
        let mut mot: super::IoUTracker<BlobBBox> = super::IoUTracker::new(5, 0.3);
        let dt = 1.0 / 25.00; // emulate 25 fps

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
        let (bboxes_one, bboxes_two, bboxes_three) = get_naive_data();
        let mut mot: super::IoUTracker<BlobBBox> = super::IoUTracker::new(5, 0.3);
        let dt = 1.0 / 25.00; // emulate 25 fps

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
}
