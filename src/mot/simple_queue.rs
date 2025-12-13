use crate::mot::blob::Blob;
use std::cmp::Ordering;
use uuid::Uuid;

// Define a tuple struct to hold distance and Blob references for priority queue ordering.
// In case of centroid metric min heap is used, in case of IoU metric max heap is used.
pub struct DistanceBlob<'a, B: Blob> {
    pub distance_metric_value: f32,
    pub min_id: Uuid,
    pub blob: &'a mut B,
}

impl<'a, B: Blob> PartialEq for DistanceBlob<'a, B> {
    fn eq(&self, other: &Self) -> bool {
        self.distance_metric_value == other.distance_metric_value
    }
}

impl<'a, B: Blob> Eq for DistanceBlob<'a, B> {}

impl<'a, B: Blob> PartialOrd for DistanceBlob<'a, B> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        other
            .distance_metric_value
            .partial_cmp(&self.distance_metric_value)
    }
}

impl<'a, B: Blob> Ord for DistanceBlob<'a, B> {
    fn cmp(&self, other: &Self) -> Ordering {
        // self.distance < other.distance
        self.partial_cmp(other).unwrap()
    }
}

mod tests {
    use crate::mot::SimpleBlob;
    #[test]
    fn test_min_heap() {
        let mut priority_queue: std::collections::BinaryHeap<super::DistanceBlob<SimpleBlob>> =
            std::collections::BinaryHeap::new();
        let blob1 = super::DistanceBlob {
            distance_metric_value: 1.0,
            min_id: uuid::Uuid::new_v4(),
            blob: &mut SimpleBlob::new(crate::utils::Rect::new(0.0, 0.0, 1.0, 1.0)),
        };
        let blob2 = super::DistanceBlob {
            distance_metric_value: 2.0,
            min_id: uuid::Uuid::new_v4(),
            blob: &mut SimpleBlob::new(crate::utils::Rect::new(0.0, 0.0, 1.0, 1.0)),
        };
        let blob3 = super::DistanceBlob {
            distance_metric_value: 3.0,
            min_id: uuid::Uuid::new_v4(),
            blob: &mut SimpleBlob::new(crate::utils::Rect::new(0.0, 0.0, 1.0, 1.0)),
        };
        let blob4 = super::DistanceBlob {
            distance_metric_value: 4.0,
            min_id: uuid::Uuid::new_v4(),
            blob: &mut SimpleBlob::new(crate::utils::Rect::new(0.0, 0.0, 1.0, 1.0)),
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

    #[test]
    fn test_max_heap() {
        let mut priority_queue: std::collections::BinaryHeap<
            std::cmp::Reverse<super::DistanceBlob<SimpleBlob>>,
        > = std::collections::BinaryHeap::new();
        let blob1 = super::DistanceBlob {
            distance_metric_value: 1.0,
            min_id: uuid::Uuid::new_v4(),
            blob: &mut SimpleBlob::new(crate::utils::Rect::new(0.0, 0.0, 1.0, 1.0)),
        };
        let blob2 = super::DistanceBlob {
            distance_metric_value: 2.0,
            min_id: uuid::Uuid::new_v4(),
            blob: &mut SimpleBlob::new(crate::utils::Rect::new(0.0, 0.0, 1.0, 1.0)),
        };
        let blob3 = super::DistanceBlob {
            distance_metric_value: 3.0,
            min_id: uuid::Uuid::new_v4(),
            blob: &mut SimpleBlob::new(crate::utils::Rect::new(0.0, 0.0, 1.0, 1.0)),
        };
        let blob4 = super::DistanceBlob {
            distance_metric_value: 4.0,
            min_id: uuid::Uuid::new_v4(),
            blob: &mut SimpleBlob::new(crate::utils::Rect::new(0.0, 0.0, 1.0, 1.0)),
        };
        priority_queue.push(std::cmp::Reverse(blob4));
        priority_queue.push(std::cmp::Reverse(blob2));
        priority_queue.push(std::cmp::Reverse(blob3));
        priority_queue.push(std::cmp::Reverse(blob1));

        assert_eq!(priority_queue.pop().unwrap().0.distance_metric_value, 4.0);
        assert_eq!(priority_queue.pop().unwrap().0.distance_metric_value, 3.0);
        assert_eq!(priority_queue.pop().unwrap().0.distance_metric_value, 2.0);
        assert_eq!(priority_queue.pop().unwrap().0.distance_metric_value, 1.0);
    }
}
