use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

use crate::mot::mot_errors;
use crate::mot::DistanceBlob;
use crate::mot::SimpleBlob;
use crate::utils::iou;
use uuid::Uuid;

/// Naive implementation of Multi-object tracker (MOT) with IoU matching
pub struct IoUTracker {
    // Max no match (max number of frames when object could not be found again). Default is 75
    max_no_match: usize,
    // IoU threshold for matching. Default is 0.3
    iou_threshold: f32,
    // Storage
    pub objects: HashMap<Uuid, SimpleBlob>,
}

impl IoUTracker {
    /// Creates default instance of IoUTracker
    ///
    /// Basic usage:
    ///
    /// ```
    /// use mot_rs::mot::IoUTracker;
    /// let mut tracker = IoUTracker::default();
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
    /// use mot_rs::mot::IoUTracker;
    /// let max_no_match: usize = 100;
    /// let iou_threshold: f32 = 0.3;
    /// let mut tracker = IoUTracker::new(max_no_match, iou_threshold);
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
        new_objects: &mut Vec<SimpleBlob>,
    ) -> Result<(), mot_errors::TrackerError> {
        for (_, object) in self.objects.iter_mut() {
            object.deactivate(); // Make sure that object is marked as deactivated
                                 // object.predict_next_position_naive(5);
            object.predict_next_position();
        }
        let mut blobs_to_register: HashMap<Uuid, SimpleBlob> = HashMap::new();

        // Add new objects to priority queue
        let mut priority_queue: BinaryHeap<Reverse<DistanceBlob>> = BinaryHeap::new();
        for new_object in new_objects.iter_mut() {
            // Find existing blob with min distance to new one
            let mut max_id = Uuid::default();
            let mut max_iou = 0.0;
            for (j, object) in self.objects.iter() {
                let iou_value = iou(&new_object.get_bbox(), &object.get_bbox());
                if iou_value > max_iou {
                    max_iou = iou_value;
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
                        v.update(&distance_blob.0.blob)?;
                        // Last but not least:
                        // We need to update ID of new object to match existing one (that is why we have &mut in function definition)
                        distance_blob.0.blob.set_id(min_id);
                        reserved_objects.insert(min_id);
                    }
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
impl fmt::Display for IoUTracker {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Maximum no match: {}\n\tIoU threshold: {}",
            self.max_no_match, self.iou_threshold
        )
    }
}

mod tests {
    use crate::utils::Rect;
    use std::collections::BinaryHeap;
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
            // vec![Rect::new(362.0,-9.0,147.0,111.0), Rect::new(3.0,103.0,236.0,346.0), Rect::new(645.0,98.0,281.0,343.0)], // here one of blobs disappears
            // vec![Rect::new(365.0,-10.0,143.0,114.0), Rect::new(645.0,99.0,283.0,345.0), Rect::new(9.0,141.0,238.0,323.0)],
        ];

        let mut mot = super::IoUTracker::new(5, 0.3);
        let dt = 1.0 / 25.00; // emulate 25 fps

        for iteration in bboxes_iterations {
            let mut blobs: Vec<super::SimpleBlob> = iteration
                .into_iter()
                .map(|bbox| super::SimpleBlob::new_with_dt(bbox, dt))
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
        let mut mot = super::IoUTracker::new(5, 0.3);
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

            // for blob in blobs.iter() {
            //     println!("id before: {:?}", blob.get_id());
            // }
            match mot.match_objects(&mut blobs) {
                Ok(_) => {}
                Err(err) => {
                    println!("{:?}", err);
                }
            };
            // for blob in blobs.iter() {
            //     println!("\tid after: {:?}", blob.get_id());
            // }
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
