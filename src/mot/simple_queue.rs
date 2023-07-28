use std::cmp::Ordering;
use crate::mot::SimpleBlob;
use uuid::Uuid;

// Define a tuple struct to hold distance and SimpleBlob references for priority queue ordering.
// In case of centroid metric min heap is used, in case of IoU metric max heap is used.
pub struct DistanceBlob<'a> {
    pub distance_metric_value: f32,
    pub min_id: Uuid,
    pub blob: &'a mut SimpleBlob,
}

impl<'a> PartialEq for DistanceBlob<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.distance_metric_value == other.distance_metric_value
    }
}

impl<'a> Eq for DistanceBlob<'a> {}

impl<'a> PartialOrd for DistanceBlob<'a> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        other.distance_metric_value.partial_cmp(&self.distance_metric_value)
    }
}

impl<'a> Ord for DistanceBlob<'a> {
    fn cmp(&self, other: &Self) -> Ordering {
        // self.distance < other.distance
        self.partial_cmp(other).unwrap()
    }
}

mod tests {
    use super::DistanceBlob;
    use crate::mot::SimpleBlob;
    use crate::utils::Rect;
    use std::collections::BinaryHeap;
    use uuid::Uuid;
    #[test]
    fn test_min_heap() {
        let mut priority_queue: BinaryHeap<DistanceBlob> = BinaryHeap::new();
        let mut blob1 = DistanceBlob {
            distance_metric_value: 1.0,
            min_id: Uuid::new_v4(),
            blob: &mut SimpleBlob::new(Rect::new(0.0, 0.0, 1.0, 1.0)),
        };
        let mut blob2 = DistanceBlob {
            distance_metric_value: 2.0,
            min_id: Uuid::new_v4(),
            blob: &mut SimpleBlob::new(Rect::new(0.0, 0.0, 1.0, 1.0)),
        };
        let mut blob3 = DistanceBlob {
            distance_metric_value: 3.0,
            min_id: Uuid::new_v4(),
            blob: &mut SimpleBlob::new(Rect::new(0.0, 0.0, 1.0, 1.0)),
        };
        let mut blob4 = DistanceBlob {
            distance_metric_value: 4.0,
            min_id: Uuid::new_v4(),
            blob: &mut SimpleBlob::new(Rect::new(0.0, 0.0, 1.0, 1.0)),
        };
        priority_queue.push(blob4);
        priority_queue.push(blob2);
        priority_queue.push(blob3);
        priority_queue.push(blob1);
        
        assert_eq!(priority_queue.pop().unwrap().distance_metric_value, 1.0);
        assert_eq!(priority_queue.pop().unwrap().distance_metric_value, 2.0);
        assert_eq!(priority_queue.pop().unwrap().distance_metric_value, 3.0);
        assert_eq!(priority_queue.pop().unwrap().distance_metric_value, 4.0);
    }

    // use super::IoUBlob;
    use std::cmp::Reverse;
    #[test]
    fn test_max_heap() {
        let mut priority_queue: BinaryHeap<Reverse<DistanceBlob>> = BinaryHeap::new();
        let mut blob1 = DistanceBlob {
            distance_metric_value: 1.0,
            min_id: Uuid::new_v4(),
            blob: &mut SimpleBlob::new(Rect::new(0.0, 0.0, 1.0, 1.0)),
        };
        let mut blob2 = DistanceBlob {
            distance_metric_value: 2.0,
            min_id: Uuid::new_v4(),
            blob: &mut SimpleBlob::new(Rect::new(0.0, 0.0, 1.0, 1.0)),
        };
        let mut blob3 = DistanceBlob {
            distance_metric_value: 3.0,
            min_id: Uuid::new_v4(),
            blob: &mut SimpleBlob::new(Rect::new(0.0, 0.0, 1.0, 1.0)),
        };
        let mut blob4 = DistanceBlob {
            distance_metric_value: 4.0,
            min_id: Uuid::new_v4(),
            blob: &mut SimpleBlob::new(Rect::new(0.0, 0.0, 1.0, 1.0)),
        };
        priority_queue.push(Reverse(blob4));
        priority_queue.push(Reverse(blob2));
        priority_queue.push(Reverse(blob3));
        priority_queue.push(Reverse(blob1));
        
        assert_eq!(priority_queue.pop().unwrap().0.distance_metric_value, 4.0);
        assert_eq!(priority_queue.pop().unwrap().0.distance_metric_value, 3.0);
        assert_eq!(priority_queue.pop().unwrap().0.distance_metric_value, 2.0);
        assert_eq!(priority_queue.pop().unwrap().0.distance_metric_value, 1.0);
    }
}