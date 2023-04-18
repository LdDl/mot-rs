use std::error::Error;
use uuid::Uuid;
use std::collections::HashMap;
use crate::mot::SimpleBlob;

/// Naive implementation of Multi-object tracker (MOT)
pub struct SimpleTracker {
    // Max no match (max number of frames when object could not be found again). Default is 75
    max_no_match: usize,
    // Threshold distance (most of time in pixels). Default 30.0
    min_dist_threshold: f32,
    // Number of points in object's track to predict next position
    depth_prediction: usize,
    // Storage
    objects: HashMap<Uuid, SimpleBlob>
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
        return SimpleTracker{
            max_no_match: 75,
            min_dist_threshold: 30.0,
            depth_prediction: 5,
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
    /// let depth_prediction: usize = 5;
    /// let mut tracker = SimpleTracker::new(max_no_match, min_dist_threshold, depth_prediction);
    /// ```
    pub fn new(_max_no_match: usize, _min_dist_threshold: f32, _depth_prediction: usize) -> Self {
        return SimpleTracker{
            max_no_match: _max_no_match,
            min_dist_threshold: _min_dist_threshold,
            depth_prediction: _depth_prediction,
            objects: HashMap::new(),
        }
    }
    // Matches new objects to existing ones
    pub fn match_objects(&mut self, new_objects: Vec<SimpleBlob>) -> Result<(), Box<dyn Error>>{
        
        for (_, object) in self.objects.iter_mut() {
            object.deactivate(); // Make sure that object is marked as deactivated
            object.predict_next_position(self.depth_prediction);
        }

        for new_object in new_objects {
            // Find existing blob with min distance to new one
            let mut min_id = Uuid::default();
            let mut min_distance = f32::MAX;
            for (j, object) in self.objects.iter() {
                let dist = new_object.distance_to(object);
                let dist_predicted = new_object.distance_to_predicted(object); // @todo: object.predicted_pos is always 0.0. Isn't it?
                let dist_verified = f32::min(dist, dist_predicted);
                if dist_verified < min_distance {
                    min_distance = dist_verified;
                    min_id = j.clone();
                }
            }
            // Additional check to filter objects
            if min_distance < new_object.get_diagonal() * 0.5 || min_distance < self.min_dist_threshold {
                match self.objects.get_mut(&min_id) {
                    Some(v) => {
                        v.update(&new_object)?;
                    },
                    None => {
                        // continue
                        panic!("immposible self.objects.get_mut(&min_id)")
                    }
                };
                continue;
            }
            // Otherwise register object as a new one
            let new_id = Uuid::new_v4();
            self.objects.insert(new_id, new_object); // Just take ownership of an object since we do not need original vector anymore
        }

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