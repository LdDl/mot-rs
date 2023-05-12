use std::cmp::Ordering;
use crate::mot::SimpleBlob;
use uuid::Uuid;

// Define a tuple struct to hold distance and SimpleBlob references for priority queue ordering
pub struct DistanceBlob<'a> {
    pub distance: f32,
    pub min_id: Uuid,
    pub blob: &'a mut SimpleBlob,
}

impl<'a> PartialEq for DistanceBlob<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl<'a> Eq for DistanceBlob<'a> {}

impl<'a> PartialOrd for DistanceBlob<'a> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        other.distance.partial_cmp(&self.distance)
    }
}

impl<'a> Ord for DistanceBlob<'a> {
    fn cmp(&self, other: &Self) -> Ordering {
        // self.distance < other.distance
        self.partial_cmp(other).unwrap()
    }
}