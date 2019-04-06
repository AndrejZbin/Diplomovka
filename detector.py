import cv2
import os

import feature_extractors as fe

from playback import *

cams = ['P1E_S1_C1', 'P1E_S1_C2']
imgListPaths = list(map(lambda c: os.path.join(c, 'all_file.txt'), cams))
print(imgListPaths)

players = list(map(lambda l: PicturePlayback([os.path.join(os.path.dirname(l), line.rstrip('\n')) for line in open(l)], 30), imgListPaths))
#players = [CameraPlayback()]
# player = [VideoPlayback('video.avi')]
#players = [YoutubePlayback('https://www.youtube.com/watch?v=3aADeK-bSMU')]


face_cascade = cv2.CascadeClassifier('opencv_data/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('opencv_data/haarcascades/haarcascade_eye.xml')

for player in players:
    player.start()


while all([player.is_playing() for player in players]):
    frames = [player.get_frame() for player in players]
    if any(list(map(lambda frame: frame is None, frames))):
        print('Ended')
        break

    grays = list(map(lambda frame: cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), frames))

    facess = list(map(lambda gray: face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    ), grays))

    frame_copies = [frame.copy() for frame in frames]


    # Draw a rectangle around the faces
    for camera_i, faces in enumerate(facess):
        for face_i, (x, y, w, h) in enumerate(faces):
            roi_gray = grays[camera_i][y:y + h, x:x + w]
            roi_color = frames[camera_i][y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)

            if len(faces) == 1 and len(eyes) == 2:
                cv2.imshow('face camera ' + str(camera_i),  cv2.resize(frame_copies[camera_i][y:y + h, x:x + w], (256, 256)))

            cv2.rectangle(frames[camera_i], (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow('Camera ' + str(camera_i), frames[camera_i])



    if cv2.waitKey(1) & 0xFF == ord('q'):
            break



cv2.destroyAllWindows()
