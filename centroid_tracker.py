from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

# original source https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/
# fixed bugs and improved for our solution


class CentroidTracker:
    next_id = 1

    def __init__(self, max_disappear=60, max_distance=50):
        self.tracks = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappear = max_disappear
        self.max_distance = max_distance

    def register(self, centroid):
        self.tracks[CentroidTracker.next_id] = centroid
        self.disappeared[CentroidTracker.next_id] = 0
        CentroidTracker.next_id += 1

    def reid(self, old, new):
        self.tracks[new] = self.tracks.get(old)
        self.disappeared[new] = self.disappeared.get(old)
        self.deregister(old)

    def deregister(self, person_id):
        self.tracks.pop(person_id)
        self.disappeared.pop(person_id)

    def update(self, rectangles, delete):
        if len(rectangles) == 0:
            to_deregister = []
            for track_id in self.disappeared.keys():
                self.disappeared[track_id] += delete
                if self.disappeared[track_id] > self.max_disappear:
                    to_deregister.append(track_id)
            for track_id in to_deregister:
                self.deregister(track_id)
            return self.tracks
        input_centroids = np.zeros((len(rectangles), 2), dtype=np.int32)

        for (i, (x1, y1, x2, y2)) in enumerate(rectangles):
            input_centroids[i] = (int((x1 + x2) / 2.0), int((y1 + y2) / 2.0))

        if len(self.tracks) == 0:
            for i in range(len(rectangles)):
                self.register(rectangles[i])
        else:
            track_ids = list(self.tracks.keys())
            centroids = np.zeros((len(track_ids), 2), dtype=np.int32)
            for (i, (x1, y1, x2, y2)) in enumerate(self.tracks.values()):
                centroids[i] = (int((x1 + x2) / 2.0), int((y1 + y2) / 2.0))

            distance_matrix = dist.cdist(np.array(centroids), input_centroids)
            rows = distance_matrix.min(axis=1).argsort()
            cols = distance_matrix.argmin(axis=1)[rows]
            used_rows = set()
            used_cols = set()

            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if distance_matrix[row, col] > self.max_distance:
                    continue

                track_id = track_ids[row]
                self.tracks[track_id] = rectangles[col]
                self.disappeared[track_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(distance_matrix.shape[0])).difference(used_rows)
            unused_cols = set(range(distance_matrix.shape[1])).difference(used_cols)

            if distance_matrix.shape[0] >= distance_matrix.shape[1]:
                for row in unused_rows:
                    track_id = track_ids[row]
                    self.disappeared[track_id] += delete

                    if self.disappeared[track_id] > self.max_disappear:
                        self.deregister(track_id)
            else:
                for col in unused_cols:
                    self.register(rectangles[col])

        filtered_dict = {k: v for (k, v) in self.tracks.items() if self.disappeared.get(k, 0) == 0}
        return filtered_dict
