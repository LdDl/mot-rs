use std::collections::{BinaryHeap, HashMap, HashSet};

use crate::mot::blob::Blob;
use crate::mot::mot_errors;
use crate::mot::DistanceBlob;
use uuid::Uuid;

/// Naive implementation of Multi-object tracker (MOT)
pub struct SimpleTracker<B: Blob> {
    // Max no match (max number of frames when object could not be found again). Default is 75
    max_no_match: usize,
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
            min_dist_threshold: _min_dist_threshold,
            objects: HashMap::new(),
        }
    }
    // Matches new objects to existing ones
    pub fn match_objects(
        &mut self,
        new_objects: &mut Vec<B>,
    ) -> Result<(), mot_errors::TrackerError> {
        for (_, object) in self.objects.iter_mut() {
            object.deactivate(); // Make sure that object is marked as deactivated
                                 // object.predict_next_position_naive(5);
            object.predict_next_position();
        }
        let mut blobs_to_register: HashMap<Uuid, B> = HashMap::new();

        // Add new objects to priority queue
        let mut priority_queue: BinaryHeap<DistanceBlob<B>> = BinaryHeap::new();
        for new_object in new_objects.iter_mut() {
            // Find existing blob with min distance to new one
            let mut min_id = Uuid::default();
            let mut min_distance = f32::MAX;
            for (j, object) in self.objects.iter() {
                let dist = new_object.distance_to(object);
                let dist_predicted = new_object.distance_to_predicted(object);
                let dist_verified = f32::min(dist, dist_predicted);
                if dist_verified < min_distance {
                    min_distance = dist_verified;
                    min_id = *j;
                }
            }
            let distance_blob = DistanceBlob {
                distance_metric_value: min_distance,
                min_id: min_id,
                blob: new_object,
            };
            priority_queue.push(distance_blob);
        }

        // We need to prevent double update of objects
        let mut reserved_objects: HashSet<Uuid> = HashSet::new();

        while let Some(distance_blob) = priority_queue.pop() {
            let min_distance = distance_blob.distance_metric_value;
            let min_id = distance_blob.min_id;

            // Check if object is already reserved
            // Since we are using priority queue with min-heap then we garantee that we will update existing objects with min distance only once.
            // For other objects with the same min_id we can create new objects
            if reserved_objects.contains(&min_id) {
                // Register it immediately and continue
                blobs_to_register.insert(distance_blob.blob.get_id(), distance_blob.blob.clone());
                continue;
            }
            // Additional check to filter objects
            if min_distance < distance_blob.blob.get_diagonal() * 0.5
                || min_distance < self.min_dist_threshold
            {
                match self.objects.get_mut(&min_id) {
                    Some(v) => {
                        v.update(&distance_blob.blob)?;
                        // Last but not least:
                        // We need to update ID of new object to match existing one (that is why we have &mut in function definition)
                        distance_blob.blob.set_id(min_id);
                        reserved_objects.insert(min_id);
                    }
                    None => {
                        return Err(mot_errors::TrackerError::from(mot_errors::NoObjectInTracker{txt: format!("immposible self.objects.get_mut(&min_id). Object ID {:?}. Distance value: {:?}", min_id, min_distance)}));
                    }
                };
            } else {
                // Otherwise register object as a new one
                blobs_to_register.insert(distance_blob.blob.get_id(), distance_blob.blob.clone());
            }
        }

        self.objects.extend(blobs_to_register);

        // Clean up existing data
        self.objects.retain(|_, object| {
            object.inc_no_match();
            // Remove object if it was not found for a long time
            let delete = object.get_no_match_times() > self.max_no_match;
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

mod tests {
    use crate::mot::blob::Blob;
    use crate::mot::test_data::{bbox_to_rect, get_naive_data, get_spread_data};
    use crate::mot::{BlobBBox, SimpleBlob};

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
}
