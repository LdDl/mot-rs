use std::error::Error;
use std::collections::{
    HashMap,
    HashSet,
    BinaryHeap,
};

use uuid::Uuid;
use crate::mot::DistanceBlob;
use crate::mot::SimpleBlob;

/// Naive implementation of Multi-object tracker (MOT)
pub struct SimpleTracker {
    // Max no match (max number of frames when object could not be found again). Default is 75
    max_no_match: usize,
    // Threshold distance (most of time in pixels). Default 30.0
    min_dist_threshold: f32,
    // Storage
    pub objects: HashMap<Uuid, SimpleBlob>
}

impl SimpleTracker {
    /// Creates default instance of SimpleTracker
    /// 
    /// Basic usage:
    /// 
    /// ```
    /// use mot_rs::mot::SimpleTracker;
    /// let mut tracker = SimpleTracker::default();
    /// ```
    pub fn default() -> Self {
        SimpleTracker{
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
    /// use mot_rs::mot::SimpleTracker;
    /// let max_no_match: usize = 100;
    /// let min_dist_threshold: f32 = 15.0;
    /// let mut tracker = SimpleTracker::new(max_no_match, min_dist_threshold);
    /// ```
    pub fn new(_max_no_match: usize, _min_dist_threshold: f32) -> Self {
        SimpleTracker{
            max_no_match: _max_no_match,
            min_dist_threshold: _min_dist_threshold,
            objects: HashMap::new(),
        }
    }
    // Matches new objects to existing ones
    pub fn match_objects(&mut self, new_objects: &mut Vec<SimpleBlob>) -> Result<(), Box<dyn Error>>{
        for (_, object) in self.objects.iter_mut() {
            object.deactivate(); // Make sure that object is marked as deactivated
            // object.predict_next_position_naive(5);
            object.predict_next_position();
        }
        let mut blobs_to_register: HashMap<Uuid, SimpleBlob> = HashMap::new();

        // Add new objects to priority queue
        let mut priority_queue: BinaryHeap<DistanceBlob> = BinaryHeap::new();
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
                distance: min_distance,
                min_id: min_id,
                blob: new_object,
            };
            priority_queue.push(distance_blob);
        }

        // We need to prevent double update of objects
        let mut reserved_objects: HashSet<Uuid> = HashSet::new();

        while let Some(mut distance_blob) = priority_queue.pop() {
            let min_distance = distance_blob.distance;
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
            if min_distance < distance_blob.blob.get_diagonal() * 0.5 || min_distance < self.min_dist_threshold {
                match self.objects.get_mut(&min_id) {
                    Some(v) => {
                        v.update(&distance_blob.blob)?;
                        // Last but not least:
                        // We need to update ID of new object to match existing one (that is why we have &mut in function definition)
                        distance_blob.blob.set_id(min_id);
                        reserved_objects.insert(min_id);
                    },
                    None => {
                        // continue
                        panic!("immposible self.objects.get_mut(&min_id)")
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
impl fmt::Display for SimpleTracker {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Maximum no match: {}\n\tMinimum threshold distance: {}",
            self.max_no_match,
            self.min_dist_threshold
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
            vec![Rect::new(378,147,173,243)],
            vec![Rect::new(374,147,180,253)],
            vec![Rect::new(375,154,178,256)],
            vec![Rect::new(376,162,177,267)],
            vec![Rect::new(375,166,178,268)],
            vec![Rect::new(375,177,186,266)],
            vec![Rect::new(370,185,197,273)],
            vec![Rect::new(363,209,203,264)],
            vec![Rect::new(70,14,227,254), Rect::new(364,214,200,262)],
            vec![Rect::new(365,218,205,263)],
            vec![Rect::new(67,23,236,246), Rect::new(366,231,209,260)],
            vec![Rect::new(73,18,227,264), Rect::new(610,47,324,355), Rect::new(370,238,199,259), Rect::new(381,-1,103,60)],
            vec![Rect::new(67,16,229,271), Rect::new(370,250,195,264), Rect::new(381,-2,106,58)],
            vec![Rect::new(62,15,233,268), Rect::new(365,257,205,264), Rect::new(379,-1,109,59)],
            vec![Rect::new(60,7,234,279), Rect::new(360,269,212,260), Rect::new(380,-1,109,60)],
            vec![Rect::new(50,41,251,295), Rect::new(619,25,308,399), Rect::new(361,276,215,265), Rect::new(380,-1,110,63)],
            vec![Rect::new(48,36,242,302), Rect::new(622,21,299,411), Rect::new(357,283,222,255), Rect::new(379,0,113,64)],
            vec![Rect::new(41,28,245,319), Rect::new(625,31,308,392), Rect::new(350,306,239,231), Rect::new(377,0,116,65)],
            vec![Rect::new(630,98,294,324), Rect::new(346,310,250,239), Rect::new(378,0,112,65)],
            vec![Rect::new(636,99,290,323), Rect::new(344,320,254,229), Rect::new(378,2,114,65)],
            vec![Rect::new(636,103,295,318), Rect::new(347,332,251,211)],
            vec![Rect::new(362,1,147,90), Rect::new(637,104,292,321), Rect::new(337,344,272,196)],
            vec![Rect::new(360,-2,152,97), Rect::new(12,74,237,324), Rect::new(639,104,293,316), Rect::new(347,350,258,185)],
            vec![Rect::new(361,-4,149,99), Rect::new(9,112,251,313), Rect::new(627,106,314,321)],
            vec![Rect::new(360,-3,151,99), Rect::new(15,115,231,311), Rect::new(633,91,297,346)],
            vec![Rect::new(362,-7,148,106), Rect::new(10,109,241,320), Rect::new(639,93,294,347)],
            vec![Rect::new(362,-9,146,109), Rect::new(12,109,233,326), Rect::new(639,95,288,347)],
            // vec![Rect::new(362,-9,147,111), Rect::new(3,103,236,346), Rect::new(645,98,281,343)], // here one of blobs disappears
            // vec![Rect::new(365,-10,143,114), Rect::new(645,99,283,345), Rect::new(9,141,238,323)],
        ];

        let mut mot = super::SimpleTracker::new(5, 15.0);
        let dt = 1.0/25.00; // emulate 25 fps

        for iteration in bboxes_iterations {
            let mut blobs: Vec<super::SimpleBlob> = iteration.into_iter().map(|bbox| {
                super::SimpleBlob::new_with_dt(bbox, dt)
            }).collect();
            match mot.match_objects(&mut blobs) {
                Ok(_) => {},
                Err(err) => {
                    println!("{:?}", err);
                }
            };
        }
        println!("id;track");
        for object in &mot.objects {
            print!("{};", object.0);
            let track = object.1.get_track();
            for (idx, pt) in track.iter().enumerate() {
                if idx == track.len() - 1 {
                    print!("{},{}", pt.x, pt.y);
                } else {
                    print!("{},{}|", pt.x, pt.y);
                }
            }
            println!();
        }
    }

    #[test]
    fn test_match_objects_naive() {
        let bboxes_one: Vec<Vec<i32>> = vec![vec![236, -25, 386, 35], vec![237, -24, 387, 36], vec![238, -22, 388, 38], vec![236, -20, 386, 40], vec![236, -19, 386, 41], vec![237, -18, 387, 42], vec![237, -18, 387, 42], vec![238, -17, 388, 43], vec![237, -14, 387, 46], vec![237, -14, 387, 46], vec![237, -12, 387, 48], vec![237, -12, 387, 48], vec![237, -11, 387, 49], vec![237, -11, 387, 49], vec![237, -10, 387, 50], vec![237, -10, 387, 50], vec![237, -8, 387, 52], vec![237, -8, 387, 52], vec![236, -7, 386, 53], vec![236, -7, 386, 53], vec![236, -6, 386, 54], vec![236, -6, 386, 54], vec![236, -2, 386, 58], vec![235, 0, 385, 60], vec![236, 2, 386, 62], vec![236, 5, 386, 65], vec![236, 9, 386, 69], vec![235, 12, 385, 72], vec![235, 14, 385, 74], vec![233, 16, 383, 76], vec![232, 26, 382, 86], vec![233, 28, 383, 88], vec![233, 40, 383, 100], vec![233, 30, 383, 90], vec![232, 22, 382, 82], vec![232, 34, 382, 94], vec![232, 21, 382, 81], vec![233, 40, 383, 100], vec![232, 40, 382, 100], vec![232, 40, 382, 100], vec![232, 36, 382, 96], vec![232, 53, 382, 113], vec![232, 50, 382, 110], vec![233, 55, 383, 115], vec![232, 50, 382, 110], vec![234, 68, 384, 128], vec![231, 49, 381, 109], vec![232, 68, 382, 128], vec![231, 31, 381, 91], vec![232, 64, 382, 124], vec![233, 71, 383, 131], vec![231, 64, 381, 124], vec![231, 74, 381, 134], vec![231, 64, 381, 124], vec![230, 77, 380, 137], vec![232, 82, 382, 142], vec![232, 78, 382, 138], vec![232, 78, 382, 138], vec![231, 79, 381, 139], vec![231, 79, 381, 139], vec![231, 91, 381, 151], vec![232, 78, 382, 138], vec![232, 78, 382, 138], vec![233, 90, 383, 150], vec![232, 92, 382, 152], vec![232, 92, 382, 152], vec![233, 98, 383, 158], vec![232, 100, 382, 160], vec![231, 92, 381, 152], vec![233, 110, 383, 170], vec![234, 92, 384, 152], vec![234, 92, 384, 152], vec![234, 110, 384, 170], vec![234, 92, 384, 152], vec![233, 104, 383, 164], vec![234, 111, 384, 171], vec![234, 106, 384, 166], vec![234, 106, 384, 166], vec![233, 124, 383, 184], vec![236, 125, 386, 185], vec![236, 125, 386, 185], vec![232, 120, 382, 180], vec![236, 131, 386, 191], vec![232, 132, 382, 192], vec![238, 139, 388, 199], vec![236, 141, 386, 201], vec![232, 151, 382, 211], vec![236, 145, 386, 205], vec![236, 145, 386, 205], vec![231, 133, 381, 193], vec![237, 148, 387, 208], vec![237, 148, 387, 208], vec![237, 148, 387, 208], vec![237, 148, 387, 208], vec![237, 148, 387, 208], vec![237, 148, 387, 208], vec![237, 148, 387, 208], vec![237, 148, 387, 208], vec![237, 148, 387, 208], vec![237, 148, 387, 208], vec![237, 148, 387, 208], vec![237, 148, 387, 208], vec![237, 148, 387, 208], vec![237, 148, 387, 208], vec![237, 148, 387, 208], vec![237, 148, 387, 208], vec![237, 148, 387, 208], vec![237, 148, 387, 208], vec![237, 148, 387, 208], vec![237, 148, 387, 208], vec![237, 148, 387, 208], vec![237, 148, 387, 208]];
        let bboxes_two: Vec<Vec<i32>> = vec![vec![321, -25, 471, 35], vec![322, -24, 472, 36], vec![323, -22, 473, 38], vec![321, -20, 471, 40], vec![321, -19, 471, 41], vec![322, -18, 472, 42], vec![322, -18, 472, 42], vec![323, -17, 473, 43], vec![322, -14, 472, 46], vec![322, -14, 472, 46], vec![322, -12, 472, 48], vec![322, -12, 472, 48], vec![322, -11, 472, 49], vec![322, -11, 472, 49], vec![322, -10, 472, 50], vec![322, -10, 472, 50], vec![322, -8, 472, 52], vec![322, -8, 472, 52], vec![321, -7, 471, 53], vec![321, -7, 471, 53], vec![321, -6, 471, 54], vec![321, -6, 471, 54], vec![321, -2, 471, 58], vec![320, 0, 470, 60], vec![321, 2, 471, 62], vec![321, 5, 471, 65], vec![321, 9, 471, 69], vec![320, 12, 470, 72], vec![320, 14, 470, 74], vec![318, 16, 468, 76], vec![317, 26, 467, 86], vec![318, 28, 468, 88], vec![318, 40, 468, 100], vec![318, 30, 468, 90], vec![317, 22, 467, 82], vec![317, 34, 467, 94], vec![317, 21, 467, 81], vec![318, 40, 468, 100], vec![317, 40, 467, 100], vec![317, 40, 467, 100], vec![317, 36, 467, 96], vec![317, 53, 467, 113], vec![317, 50, 467, 110], vec![318, 55, 468, 115], vec![317, 50, 467, 110], vec![319, 68, 469, 128], vec![316, 49, 466, 109], vec![317, 68, 467, 128], vec![316, 31, 466, 91], vec![317, 64, 467, 124], vec![318, 71, 468, 131], vec![316, 64, 466, 124], vec![316, 74, 466, 134], vec![316, 64, 466, 124], vec![315, 77, 465, 137], vec![317, 82, 467, 142], vec![317, 78, 467, 138], vec![317, 78, 467, 138], vec![316, 79, 466, 139], vec![316, 79, 466, 139], vec![316, 91, 466, 151], vec![317, 78, 467, 138], vec![317, 78, 467, 138], vec![318, 90, 468, 150], vec![317, 92, 467, 152], vec![317, 92, 467, 152], vec![318, 98, 468, 158], vec![317, 100, 467, 160], vec![316, 92, 466, 152], vec![318, 110, 468, 170], vec![319, 92, 469, 152], vec![319, 92, 469, 152], vec![319, 110, 469, 170], vec![319, 92, 469, 152], vec![318, 104, 468, 164], vec![319, 111, 469, 171], vec![319, 106, 469, 166], vec![319, 106, 469, 166], vec![318, 124, 468, 184], vec![321, 125, 471, 185], vec![321, 125, 471, 185], vec![317, 120, 467, 180], vec![321, 131, 471, 191], vec![317, 132, 467, 192], vec![323, 139, 473, 199], vec![321, 141, 471, 201], vec![317, 151, 467, 211], vec![321, 145, 471, 205], vec![321, 145, 471, 205], vec![316, 133, 466, 193], vec![322, 148, 472, 208], vec![322, 148, 472, 208], vec![322, 148, 472, 208], vec![322, 148, 472, 208], vec![322, 148, 472, 208], vec![322, 148, 472, 208], vec![322, 148, 472, 208], vec![322, 148, 472, 208], vec![322, 148, 472, 208], vec![322, 148, 472, 208], vec![322, 148, 472, 208], vec![322, 148, 472, 208], vec![322, 148, 472, 208], vec![322, 148, 472, 208], vec![322, 148, 472, 208], vec![322, 148, 472, 208], vec![322, 148, 472, 208], vec![322, 148, 472, 208], vec![322, 148, 472, 208], vec![322, 148, 472, 208], vec![322, 148, 472, 208], vec![322, 148, 472, 208]];
        let bboxes_three: Vec<Vec<i32>> = vec![vec![151, -25, 301, 35], vec![152, -24, 302, 36], vec![153, -22, 303, 38], vec![151, -20, 301, 40], vec![151, -19, 301, 41], vec![152, -18, 302, 42], vec![152, -18, 302, 42], vec![153, -17, 303, 43], vec![152, -14, 302, 46], vec![152, -14, 302, 46], vec![152, -12, 302, 48], vec![152, -12, 302, 48], vec![152, -11, 302, 49], vec![152, -11, 302, 49], vec![152, -10, 302, 50], vec![152, -10, 302, 50], vec![152, -8, 302, 52], vec![152, -8, 302, 52], vec![151, -7, 301, 53], vec![151, -7, 301, 53], vec![151, -6, 301, 54], vec![151, -6, 301, 54], vec![151, -2, 301, 58], vec![150, 0, 300, 60], vec![151, 2, 301, 62], vec![151, 5, 301, 65], vec![151, 9, 301, 69], vec![150, 12, 300, 72], vec![150, 14, 300, 74], vec![148, 16, 298, 76], vec![147, 26, 297, 86], vec![148, 28, 298, 88], vec![148, 40, 298, 100], vec![148, 30, 298, 90], vec![147, 22, 297, 82], vec![147, 34, 297, 94], vec![147, 21, 297, 81], vec![148, 40, 298, 100], vec![147, 40, 297, 100], vec![147, 40, 297, 100], vec![147, 36, 297, 96], vec![147, 53, 297, 113], vec![147, 50, 297, 110], vec![148, 55, 298, 115], vec![147, 50, 297, 110], vec![149, 68, 299, 128], vec![146, 49, 296, 109], vec![147, 68, 297, 128], vec![146, 31, 296, 91], vec![147, 64, 297, 124], vec![148, 71, 298, 131], vec![146, 64, 296, 124], vec![146, 74, 296, 134], vec![146, 64, 296, 124], vec![145, 77, 295, 137], vec![147, 82, 297, 142], vec![147, 78, 297, 138], vec![147, 78, 297, 138], vec![146, 79, 296, 139], vec![146, 79, 296, 139], vec![146, 91, 296, 151], vec![147, 78, 297, 138], vec![147, 78, 297, 138], vec![148, 90, 298, 150], vec![147, 92, 297, 152], vec![147, 92, 297, 152], vec![148, 98, 298, 158], vec![147, 100, 297, 160], vec![146, 92, 296, 152], vec![148, 110, 298, 170], vec![149, 92, 299, 152], vec![149, 92, 299, 152], vec![149, 110, 299, 170], vec![149, 92, 299, 152], vec![148, 104, 298, 164], vec![149, 111, 299, 171], vec![149, 106, 299, 166], vec![149, 106, 299, 166], vec![148, 124, 298, 184], vec![151, 125, 301, 185], vec![151, 125, 301, 185], vec![147, 120, 297, 180], vec![151, 131, 301, 191], vec![147, 132, 297, 192], vec![153, 139, 303, 199], vec![151, 141, 301, 201], vec![147, 151, 297, 211], vec![151, 145, 301, 205], vec![151, 145, 301, 205], vec![146, 133, 296, 193], vec![152, 148, 302, 208], vec![152, 148, 302, 208], vec![152, 148, 302, 208], vec![152, 148, 302, 208], vec![152, 148, 302, 208], vec![152, 148, 302, 208], vec![152, 148, 302, 208], vec![152, 148, 302, 208], vec![152, 148, 302, 208], vec![152, 148, 302, 208], vec![152, 148, 302, 208], vec![152, 148, 302, 208], vec![152, 148, 302, 208], vec![152, 148, 302, 208], vec![152, 148, 302, 208], vec![152, 148, 302, 208], vec![152, 148, 302, 208], vec![152, 148, 302, 208], vec![152, 148, 302, 208], vec![152, 148, 302, 208], vec![152, 148, 302, 208], vec![152, 148, 302, 208]];
        let mut mot = super::SimpleTracker::new(5, 15.0);
        let dt = 1.0/25.00; // emulate 25 fps

        for (bbox_one, bbox_two, bbox_three) in itertools::izip!(bboxes_one, bboxes_two, bboxes_three) {
            let blob_one = super::SimpleBlob::new_with_dt(Rect::new(bbox_one[0], bbox_one[1], bbox_one[2]-bbox_one[0], bbox_one[3]-bbox_one[1]), dt);
            let blob_two = super::SimpleBlob::new_with_dt(Rect::new(bbox_two[0],bbox_two[1],bbox_two[2] -bbox_two[0],bbox_two[3]- bbox_two[1]), dt);
            let blob_three = super::SimpleBlob::new_with_dt(Rect::new(bbox_three[0],bbox_three[1],bbox_three[2] -bbox_three[0],bbox_three[3]- bbox_three[1]), dt);

            let mut blobs = vec![blob_one, blob_two, blob_three];

            // for blob in blobs.iter() {
            //     println!("id before: {:?}", blob.get_id());
            // }
            match mot.match_objects(&mut blobs) {
                Ok(_) => {},
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