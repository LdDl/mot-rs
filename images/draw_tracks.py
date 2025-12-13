import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random as r

dict_plots = {
    'simple_blob_spread': {
        'title': 'MOT for SimpleBlob\nvia simple tracker for spread detections',
        'xlim': [100, 800],
        'out': 'images/mot_simple_spread.png'
    },
    'simple_blob_naive': {
        'title': 'MOT for SimpleBlob\nvia simple tracker for naive detections',
        'xlim': [200, 420],
        'out': 'images/mot_simple_naive.png'
    },
    'simple_blob_iou_spread': {
        'title': 'MOT for SimpleBlob\nvia IoU tracker for spread detections',
        'xlim': [100, 800],
        'out': 'images/mot_simple_iou_spread.png'
    },
    'simple_blob_iou_naive': {
        'title': 'MOT for SimpleBlob\nvia IoU tracker for naive detections',
        'xlim': [200, 420],
        'out': 'images/mot_simple_iou_naive.png'
    },
    'simple_blob_bytetrack_spread': {
        'title': 'MOT for SimpleBlob\nvia ByteTrack for spread detections',
        'xlim': [100, 800],
        'out': 'images/mot_simple_bytetrack_spread.png'
    },
    'simple_blob_bytetrack_naive': {
        'title': 'MOT for SimpleBlob\nvia ByteTrack for naive detections',
        'xlim': [200, 420],
        'out': 'images/mot_simple_bytetrack_naive.png'
    },

    'bbox_blob_spread': {
        'title': 'MOT for BlobBBox\nvia simple tracker for spread detections',
        'xlim': [100, 800],
        'out': 'images/mot_bbox_spread.png'
    },
    'bbox_blob_naive': {
        'title': 'MOT for BlobBBox\nvia simple tracker for naive detections',
        'xlim': [200, 420],
        'out': 'images/mot_bbox_naive.png'
    },
    'bbox_blob_iou_spread': {
        'title': 'MOT for BlobBBox\nvia IoU tracker for spread detections',
        'xlim': [100, 800],
        'out': 'images/mot_bbox_iou_spread.png'
    },
    'bbox_blob_iou_naive': {
        'title': 'MOT for BlobBBox\nvia IoU tracker for naive detections',
        'xlim': [200, 420],
        'out': 'images/mot_bbox_iou_naive.png'
    },
    'bbox_blob_bytetrack_spread': {
        'title': 'MOT for BlobBBox\nvia ByteTrack for spread detections',
        'xlim': [100, 800],
        'out': 'images/mot_bbox_bytetrack_spread.png'
    },
    'bbox_blob_bytetrack_naive': {
        'title': 'MOT for BlobBBox\nvia ByteTrack for naive detections',
        'xlim': [200, 420],
        'out': 'images/mot_bbox_bytetrack_naive.png'
    }
}

selected_plot = 'bbox_blob_bytetrack_naive'

blobs = []
has_bbox_data = False

with open('images/blobs.csv', 'r') as f:
    next(f) # skip header
    lineCounter = 0
    for line in f:
        lineCounter += 1
        data = line.rstrip().split(';')
        blob = {'id': str(lineCounter), 'x': [], 'y': [], 'w': [], 'h': []}
        track_data = data[1].split('|')
        for point_data in track_data:
            point = point_data.split(',')
            blob['x'].append(float(point[0]))
            blob['y'].append(float(point[1]))
            # Check if we have bbox data (x, y, w, h)
            if len(point) >= 4:
                blob['w'].append(float(point[2]))
                blob['h'].append(float(point[3]))
                has_bbox_data = True
        blobs.append(blob)

dpi = 100

if has_bbox_data:
    # Create 2x1 subplot: left for trajectories, right for bboxes at selected frames
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(1440/dpi, 480/dpi), dpi=dpi)

    # Generate colors for each blob
    blob_colors = []
    for blob in blobs:
        hex_color = '#' + ''.join([r.choice('0123456789ABCDEF') for _ in range(6)])
        blob_colors.append(hex_color)

    # Left plot: Center trajectories
    for i, blob in enumerate(blobs):
        ax1.plot(blob['x'], blob['y'], color=blob_colors[i], label=('Blob #' + blob['id']))

    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.2)
    ax1.set_xlim(dict_plots[selected_plot]['xlim'])
    ax1.invert_yaxis()
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title("Center Trajectories", fontsize=12, fontweight='bold')

    # Right plot: Bounding boxes at selected frames
    n_frames = len(blobs[0]['x']) if blobs else 0
    step = max(1, n_frames // 8)  # Show ~8 frames
    frame_indices = list(range(0, n_frames, step))

    for i, blob in enumerate(blobs):
        color = blob_colors[i]
        # Draw trajectory line (faint)
        ax2.plot(blob['x'], blob['y'], color=color, alpha=0.3, linewidth=1)

        # Draw bboxes at selected frames
        for frame in frame_indices:
            if frame < len(blob['x']) and frame < len(blob['w']):
                cx, cy = blob['x'][frame], blob['y'][frame]
                w, h = blob['w'][frame], blob['h'][frame]
                # Convert center to top-left corner
                rect = patches.Rectangle(
                    (cx - w/2, cy - h/2), w, h,
                    linewidth=1.5, edgecolor=color, facecolor='none',
                    alpha=0.3 + 0.7 * frame / n_frames,  # Fade in over time
                    label=('Blob #' + blob['id']) if frame == frame_indices[0] else None
                )
                ax2.add_patch(rect)
                ax2.plot(cx, cy, 'o', color=color, markersize=3, alpha=0.5)

    ax2.legend(loc="upper left")
    ax2.grid(alpha=0.2)
    ax2.set_xlim(dict_plots[selected_plot]['xlim'])
    ax2.invert_yaxis()
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title("Bounding Boxes (selected frames)", fontsize=12, fontweight='bold')
    ax2.set_aspect('equal', adjustable='datalim')
    ax2.autoscale()

    plt.suptitle(dict_plots[selected_plot]['title'], fontsize=15, fontweight='bold')
    plt.tight_layout()
else:
    # Simple centroid-only plot (original behavior)
    plt.figure(figsize=(720/dpi, 480/dpi), dpi=dpi)
    for blob in blobs:
        hex_color = '#' + ''.join([r.choice('0123456789ABCDEF') for _ in range(6)])
        plt.plot(blob['x'], blob['y'], color=hex_color, label=('Blob #' + blob['id']))

    plt.legend(loc="upper left")
    plt.grid(alpha=0.2)
    plt.xlim(dict_plots[selected_plot]['xlim'])
    ax = plt.gca()
    ax.invert_yaxis()

    plt.title(dict_plots[selected_plot]['title'], fontsize=15, fontweight='bold')

plt.savefig(dict_plots[selected_plot]['out'], dpi=dpi)
plt.show()
