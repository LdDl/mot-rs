import matplotlib.pyplot as plt
import random as r

dict_plots = {
    'simple_blob_naive_spread': {
        'title': 'MOT for SimpleBlob\nvia simple tracker for spread detections',
        'xlim': [100, 800],
        'out': 'images/mot_simple_spread.png'
    },
    'simple_blob_naive_naive': {
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
    'bytetrack_blob_spread': {
        'title': 'MOT for SimpleBlob\nvia ByteTrack for spread detections',
        'xlim': [100, 800],
        'out': 'images/mot_simple_bytetrack_spread.png'
    },
    'bytetrack_blob_naive': {
        'title': 'MOT for SimpleBlob\nvia ByteTrack for naive detections',
        'xlim': [200, 420],
        'out': 'images/mot_simple_bytetrack_naive.png'
    }
}

selected_plot = 'bytetrack_blob_spread'

blobs = []

with open('images/blobs.csv', 'r') as f:
    next(f) # skip header
    lineCounter = 0
    for line in f:
        lineCounter += 1
        data = line.rstrip().split(';')
        blob = {'id': str(lineCounter), 'x': [], 'y': []}
        track_data = data[1].split('|')
        for point_data in track_data:
            point = point_data.split(',')
            blob['x'].append(float(point[0]))
            blob['y'].append(float(point[1]))
        blobs.append(blob)

dpi = 100
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