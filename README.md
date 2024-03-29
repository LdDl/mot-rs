# Very naive implementation of Multi-object tracking in Rust programming language

## Table of Contents

- [About](#about)
- [How to use](#how-to-use)
- [References](#References)

## About

**Why is it even exists when there are a lot of other solutions (better perfomance, better logic, better overall and etc.)?**

It is simple as that: 
> _What I cannot create, I do not understand_
> Richard Feynman

**What is this good for?**

You can use this library to track vehicles / peoples and etc. when you don't need that much accuracy or ReID.

**Are more advanced algorithms considered to be implemented in futher?**

Yes, I do think so. I guess that [SORT](https://arxiv.org/abs/1602.00763) will be the next one.

If you want to you can contribute via opening [Pull Request](https://github.com/LdDl/mot-rs/compare)

I've implemented two metrics to match objects:
* [Centroids distance + diagonal](src/mot/simple.rs#L56)
* [IoU](src/mot/iou_tracker.rs#L58)

## How to use

Add dependency to your Cargo.toml file

```toml
[package]
....

[dependencies]
...
mot-rs = "0.1.0"
...
```

Let's create really synthetic example and define similar trajectories for three objects.
We're using pretty simple MOT algorithm, so no hard tasks for now.
We are going to use [itertools](https://github.com/rust-itertools/itertools#itertools) just to simplify zipping trajectories.

```rust
use mot_rs::mot::{
    SimpleTracker,
    SimpleBlob
};
use mot_rs::utils::{
    Rect
};

fn main() {
    let bboxes_one: Vec<Vec<f32>> = vec![vec![236, -25, 386, 35], vec![237, -24, 387, 36], vec![238, -22, 388, 38], vec![236, -20, 386, 40], vec![236, -19, 386, 41], vec![237, -18, 387, 42], vec![237, -18, 387, 42], vec![238, -17, 388, 43], vec![237, -14, 387, 46], vec![237, -14, 387, 46], vec![237, -12, 387, 48], vec![237, -12, 387, 48], vec![237, -11, 387, 49], vec![237, -11, 387, 49], vec![237, -10, 387, 50], vec![237, -10, 387, 50], vec![237, -8, 387, 52], vec![237, -8, 387, 52], vec![236, -7, 386, 53], vec![236, -7, 386, 53], vec![236, -6, 386, 54], vec![236, -6, 386, 54], vec![236, -2, 386, 58], vec![235, 0, 385, 60], vec![236, 2, 386, 62], vec![236, 5, 386, 65], vec![236, 9, 386, 69], vec![235, 12, 385, 72], vec![235, 14, 385, 74], vec![233, 16, 383, 76], vec![232, 26, 382, 86], vec![233, 28, 383, 88], vec![233, 40, 383, 100], vec![233, 30, 383, 90], vec![232, 22, 382, 82], vec![232, 34, 382, 94], vec![232, 21, 382, 81], vec![233, 40, 383, 100], vec![232, 40, 382, 100], vec![232, 40, 382, 100], vec![232, 36, 382, 96], vec![232, 53, 382, 113], vec![232, 50, 382, 110], vec![233, 55, 383, 115], vec![232, 50, 382, 110], vec![234, 68, 384, 128], vec![231, 49, 381, 109], vec![232, 68, 382, 128], vec![231, 31, 381, 91], vec![232, 64, 382, 124], vec![233, 71, 383, 131], vec![231, 64, 381, 124], vec![231, 74, 381, 134], vec![231, 64, 381, 124], vec![230, 77, 380, 137], vec![232, 82, 382, 142], vec![232, 78, 382, 138], vec![232, 78, 382, 138], vec![231, 79, 381, 139], vec![231, 79, 381, 139], vec![231, 91, 381, 151], vec![232, 78, 382, 138], vec![232, 78, 382, 138], vec![233, 90, 383, 150], vec![232, 92, 382, 152], vec![232, 92, 382, 152], vec![233, 98, 383, 158], vec![232, 100, 382, 160], vec![231, 92, 381, 152], vec![233, 110, 383, 170], vec![234, 92, 384, 152], vec![234, 92, 384, 152], vec![234, 110, 384, 170], vec![234, 92, 384, 152], vec![233, 104, 383, 164], vec![234, 111, 384, 171], vec![234, 106, 384, 166], vec![234, 106, 384, 166], vec![233, 124, 383, 184], vec![236, 125, 386, 185], vec![236, 125, 386, 185], vec![232, 120, 382, 180], vec![236, 131, 386, 191], vec![232, 132, 382, 192], vec![238, 139, 388, 199], vec![236, 141, 386, 201], vec![232, 151, 382, 211], vec![236, 145, 386, 205], vec![236, 145, 386, 205], vec![231, 133, 381, 193], vec![237, 148, 387, 208], vec![237, 148, 387, 208], vec![237, 148, 387, 208], vec![237, 148, 387, 208], vec![237, 148, 387, 208], vec![237, 148, 387, 208], vec![237, 148, 387, 208], vec![237, 148, 387, 208], vec![237, 148, 387, 208], vec![237, 148, 387, 208], vec![237, 148, 387, 208], vec![237, 148, 387, 208], vec![237, 148, 387, 208], vec![237, 148, 387, 208], vec![237, 148, 387, 208], vec![237, 148, 387, 208], vec![237, 148, 387, 208], vec![237, 148, 387, 208], vec![237, 148, 387, 208], vec![237, 148, 387, 208], vec![237, 148, 387, 208], vec![237, 148, 387, 208]];
    let bboxes_two: Vec<Vec<f32>> = vec![vec![321, -25, 471, 35], vec![322, -24, 472, 36], vec![323, -22, 473, 38], vec![321, -20, 471, 40], vec![321, -19, 471, 41], vec![322, -18, 472, 42], vec![322, -18, 472, 42], vec![323, -17, 473, 43], vec![322, -14, 472, 46], vec![322, -14, 472, 46], vec![322, -12, 472, 48], vec![322, -12, 472, 48], vec![322, -11, 472, 49], vec![322, -11, 472, 49], vec![322, -10, 472, 50], vec![322, -10, 472, 50], vec![322, -8, 472, 52], vec![322, -8, 472, 52], vec![321, -7, 471, 53], vec![321, -7, 471, 53], vec![321, -6, 471, 54], vec![321, -6, 471, 54], vec![321, -2, 471, 58], vec![320, 0, 470, 60], vec![321, 2, 471, 62], vec![321, 5, 471, 65], vec![321, 9, 471, 69], vec![320, 12, 470, 72], vec![320, 14, 470, 74], vec![318, 16, 468, 76], vec![317, 26, 467, 86], vec![318, 28, 468, 88], vec![318, 40, 468, 100], vec![318, 30, 468, 90], vec![317, 22, 467, 82], vec![317, 34, 467, 94], vec![317, 21, 467, 81], vec![318, 40, 468, 100], vec![317, 40, 467, 100], vec![317, 40, 467, 100], vec![317, 36, 467, 96], vec![317, 53, 467, 113], vec![317, 50, 467, 110], vec![318, 55, 468, 115], vec![317, 50, 467, 110], vec![319, 68, 469, 128], vec![316, 49, 466, 109], vec![317, 68, 467, 128], vec![316, 31, 466, 91], vec![317, 64, 467, 124], vec![318, 71, 468, 131], vec![316, 64, 466, 124], vec![316, 74, 466, 134], vec![316, 64, 466, 124], vec![315, 77, 465, 137], vec![317, 82, 467, 142], vec![317, 78, 467, 138], vec![317, 78, 467, 138], vec![316, 79, 466, 139], vec![316, 79, 466, 139], vec![316, 91, 466, 151], vec![317, 78, 467, 138], vec![317, 78, 467, 138], vec![318, 90, 468, 150], vec![317, 92, 467, 152], vec![317, 92, 467, 152], vec![318, 98, 468, 158], vec![317, 100, 467, 160], vec![316, 92, 466, 152], vec![318, 110, 468, 170], vec![319, 92, 469, 152], vec![319, 92, 469, 152], vec![319, 110, 469, 170], vec![319, 92, 469, 152], vec![318, 104, 468, 164], vec![319, 111, 469, 171], vec![319, 106, 469, 166], vec![319, 106, 469, 166], vec![318, 124, 468, 184], vec![321, 125, 471, 185], vec![321, 125, 471, 185], vec![317, 120, 467, 180], vec![321, 131, 471, 191], vec![317, 132, 467, 192], vec![323, 139, 473, 199], vec![321, 141, 471, 201], vec![317, 151, 467, 211], vec![321, 145, 471, 205], vec![321, 145, 471, 205], vec![316, 133, 466, 193], vec![322, 148, 472, 208], vec![322, 148, 472, 208], vec![322, 148, 472, 208], vec![322, 148, 472, 208], vec![322, 148, 472, 208], vec![322, 148, 472, 208], vec![322, 148, 472, 208], vec![322, 148, 472, 208], vec![322, 148, 472, 208], vec![322, 148, 472, 208], vec![322, 148, 472, 208], vec![322, 148, 472, 208], vec![322, 148, 472, 208], vec![322, 148, 472, 208], vec![322, 148, 472, 208], vec![322, 148, 472, 208], vec![322, 148, 472, 208], vec![322, 148, 472, 208], vec![322, 148, 472, 208], vec![322, 148, 472, 208], vec![322, 148, 472, 208], vec![322, 148, 472, 208]];
    let bboxes_three: Vec<Vec<f32>> = vec![vec![151, -25, 301, 35], vec![152, -24, 302, 36], vec![153, -22, 303, 38], vec![151, -20, 301, 40], vec![151, -19, 301, 41], vec![152, -18, 302, 42], vec![152, -18, 302, 42], vec![153, -17, 303, 43], vec![152, -14, 302, 46], vec![152, -14, 302, 46], vec![152, -12, 302, 48], vec![152, -12, 302, 48], vec![152, -11, 302, 49], vec![152, -11, 302, 49], vec![152, -10, 302, 50], vec![152, -10, 302, 50], vec![152, -8, 302, 52], vec![152, -8, 302, 52], vec![151, -7, 301, 53], vec![151, -7, 301, 53], vec![151, -6, 301, 54], vec![151, -6, 301, 54], vec![151, -2, 301, 58], vec![150, 0, 300, 60], vec![151, 2, 301, 62], vec![151, 5, 301, 65], vec![151, 9, 301, 69], vec![150, 12, 300, 72], vec![150, 14, 300, 74], vec![148, 16, 298, 76], vec![147, 26, 297, 86], vec![148, 28, 298, 88], vec![148, 40, 298, 100], vec![148, 30, 298, 90], vec![147, 22, 297, 82], vec![147, 34, 297, 94], vec![147, 21, 297, 81], vec![148, 40, 298, 100], vec![147, 40, 297, 100], vec![147, 40, 297, 100], vec![147, 36, 297, 96], vec![147, 53, 297, 113], vec![147, 50, 297, 110], vec![148, 55, 298, 115], vec![147, 50, 297, 110], vec![149, 68, 299, 128], vec![146, 49, 296, 109], vec![147, 68, 297, 128], vec![146, 31, 296, 91], vec![147, 64, 297, 124], vec![148, 71, 298, 131], vec![146, 64, 296, 124], vec![146, 74, 296, 134], vec![146, 64, 296, 124], vec![145, 77, 295, 137], vec![147, 82, 297, 142], vec![147, 78, 297, 138], vec![147, 78, 297, 138], vec![146, 79, 296, 139], vec![146, 79, 296, 139], vec![146, 91, 296, 151], vec![147, 78, 297, 138], vec![147, 78, 297, 138], vec![148, 90, 298, 150], vec![147, 92, 297, 152], vec![147, 92, 297, 152], vec![148, 98, 298, 158], vec![147, 100, 297, 160], vec![146, 92, 296, 152], vec![148, 110, 298, 170], vec![149, 92, 299, 152], vec![149, 92, 299, 152], vec![149, 110, 299, 170], vec![149, 92, 299, 152], vec![148, 104, 298, 164], vec![149, 111, 299, 171], vec![149, 106, 299, 166], vec![149, 106, 299, 166], vec![148, 124, 298, 184], vec![151, 125, 301, 185], vec![151, 125, 301, 185], vec![147, 120, 297, 180], vec![151, 131, 301, 191], vec![147, 132, 297, 192], vec![153, 139, 303, 199], vec![151, 141, 301, 201], vec![147, 151, 297, 211], vec![151, 145, 301, 205], vec![151, 145, 301, 205], vec![146, 133, 296, 193], vec![152, 148, 302, 208], vec![152, 148, 302, 208], vec![152, 148, 302, 208], vec![152, 148, 302, 208], vec![152, 148, 302, 208], vec![152, 148, 302, 208], vec![152, 148, 302, 208], vec![152, 148, 302, 208], vec![152, 148, 302, 208], vec![152, 148, 302, 208], vec![152, 148, 302, 208], vec![152, 148, 302, 208], vec![152, 148, 302, 208], vec![152, 148, 302, 208], vec![152, 148, 302, 208], vec![152, 148, 302, 208], vec![152, 148, 302, 208], vec![152, 148, 302, 208], vec![152, 148, 302, 208], vec![152, 148, 302, 208], vec![152, 148, 302, 208], vec![152, 148, 302, 208]];
    let mut mot = SimpleTracker::new(5, 15.0);
    let dt = 1.0/25.00; // emulate 25 fps

    for (bbox_one, bbox_two, bbox_three) in itertools::izip!(bboxes_one, bboxes_two, bboxes_three) {
        let blob_one = SimpleBlob::new_with_dt(Rect::new(bbox_one[0], bbox_one[1], bbox_one[2]-bbox_one[0], bbox_one[3]-bbox_one[1]), dt);
        let blob_two = SimpleBlob::new_with_dt(Rect::new(bbox_two[0],bbox_two[1],bbox_two[2] -bbox_two[0],bbox_two[3]- bbox_two[1]), dt);
        let blob_three = SimpleBlob::new_with_dt(Rect::new(bbox_three[0],bbox_three[1],bbox_three[2] -bbox_three[0],bbox_three[3]- bbox_three[1]), dt);

        match mot.match_objects(vec![blob_one, blob_two, blob_three]) {
            Ok(_) => {},
            Err(err) => {
                println!("{:?}", err);
            }
        };

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
}

```

If we plot results on a single image we should get something like:

<img src="images/mot_simple_naive.png" width="720">

## References
- [Implementation of Kalman filter, Dimitrii Lopanov, 2023](https://github.com/LdDl/kalman-rs#implementation-of-discrete-kalman-filter-for-object-tracking-purposes)
- [Wikipedia](https://en.wikipedia.org/wiki/Multiple_object_tracking)
