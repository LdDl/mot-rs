use std::cmp::Ordering;
use std::collections::BinaryHeap;

struct PendingMatch {
    detection: usize,
    track: usize,
    cost: f32,
}

impl PartialEq for PendingMatch {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost
    }
}

impl Eq for PendingMatch {}

impl PartialOrd for PendingMatch {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        other.cost.partial_cmp(&self.cost)
    }
}

impl Ord for PendingMatch {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

// Each row contains acceptable (track index, cost) pairs for one detection.
// Lower costs win. Track indices follow the original iteration order for ties.
pub fn greedy_matching(
    mut candidates: Vec<Vec<(usize, f32)>>,
    num_tracks: usize,
) -> Vec<(usize, usize)> {
    let mut pending = BinaryHeap::with_capacity(candidates.len());
    for (detection, row) in candidates.iter_mut().enumerate() {
        if row.is_empty() {
            continue;
        }
        let mut best = 0;
        for index in 1..row.len() {
            if row[index].1 < row[best].1 {
                best = index;
            }
        }
        let (track, cost) = row.swap_remove(best);
        pending.push(PendingMatch {
            detection,
            track,
            cost,
        });
    }

    let mut reserved = vec![false; num_tracks];
    let mut sorted = vec![false; candidates.len()];
    let mut matches = Vec::with_capacity(num_tracks.min(candidates.len()));
    while let Some(mut candidate) = pending.pop() {
        if reserved[candidate.track] {
            let row = &mut candidates[candidate.detection];
            if !sorted[candidate.detection] {
                // Sort only after a conflict, with the best remaining candidate at the end.
                row.sort_unstable_by(|a, b| {
                    b.1.partial_cmp(&a.1).unwrap().then_with(|| b.0.cmp(&a.0))
                });
                sorted[candidate.detection] = true;
            }
            while let Some((track, cost)) = row.pop() {
                if !reserved[track] {
                    candidate.track = track;
                    candidate.cost = cost;
                    // A fallback must compete with other detections at its own cost.
                    pending.push(candidate);
                    break;
                }
            }
        } else {
            reserved[candidate.track] = true;
            matches.push((candidate.detection, candidate.track));
        }
    }
    matches
}

#[cfg(test)]
mod tests {
    use super::greedy_matching;

    #[test]
    fn matches_global_greedy_reference() {
        for num_tracks in [0, 1, 2, 7, 31] {
            for num_detections in [0, 1, 3, 11, 32] {
                let count = num_tracks * num_detections;
                let increasing: Vec<_> = (0..count).collect();
                let decreasing: Vec<_> = (0..count).rev().collect();
                // Shift the preferred track between detections while keeping all costs unique.
                let shifted_columns: Vec<_> = (0..num_detections)
                    .flat_map(|detection| {
                        (0..num_tracks).map(move |track| {
                            ((track + detection) % num_tracks) * num_detections + detection
                        })
                    })
                    .collect();
                for costs in [increasing, decreasing, shifted_columns] {
                    for divisor in [1, 3] {
                        let mut candidates = vec![vec![]; num_detections];
                        let mut pairs = Vec::new();
                        for detection in 0..num_detections {
                            for track in 0..num_tracks {
                                let cost = costs[detection * num_tracks + track];
                                if cost % divisor == 0 {
                                    candidates[detection].push((track, cost as f32));
                                    pairs.push((cost, detection, track));
                                }
                            }
                        }
                        pairs.sort_unstable();
                        let mut used_detections = vec![false; num_detections];
                        let mut used_tracks = vec![false; num_tracks];
                        let mut expected = Vec::new();
                        for (_, detection, track) in pairs {
                            if !used_detections[detection] && !used_tracks[track] {
                                used_detections[detection] = true;
                                used_tracks[track] = true;
                                expected.push((detection, track));
                            }
                        }
                        assert_eq!(greedy_matching(candidates, num_tracks), expected);
                    }
                }
            }
        }
    }

    #[test]
    fn equal_cost_alternatives_follow_track_order() {
        let candidates = vec![vec![(0, 0.0)], vec![(0, 1.0), (1, 2.0), (2, 2.0)], vec![]];
        assert_eq!(greedy_matching(candidates, 3), vec![(0, 0), (1, 1)]);
    }
}
