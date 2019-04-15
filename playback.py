import time
import cv2
import pafy


class Playback:
    def start(self):
        raise NotImplemented()

    def stop(self):
        raise NotImplemented()

    def get_frame(self):
        raise NotImplemented()

    def is_playing(self):
        raise NotImplemented()


class PicturePlayback(Playback):
    def __init__(self, captures, framerate=30, each_frame=True):
        self.captures = captures
        self.framerate = framerate
        self.frametime = (1000/self.framerate)
        self.start_time = None
        self.previous_index = -1
        self.frame = None
        self.each_frame = each_frame

    def start(self):
        self.start_time = int(round(time.time() * 1000))

    def stop(self):
        self.start_time = None

    def get_frame(self):
        time_diff = (int(round(time.time() * 1000)) - self.start_time)
        frame_index = int(time_diff/self.frametime)
        if self.each_frame and frame_index > self.previous_index + 1:
            frame_index = self.previous_index + 1
        if frame_index != self.previous_index:
            if frame_index >= len(self.captures):
                self.stop()
            else:
                self.previous_index = frame_index
                self.frame = cv2.imread(self.captures[frame_index], cv2.IMREAD_COLOR)
        return self.frame

    def is_playing(self):
        return self.start_time is not None


class CameraPlayback(Playback):
    def __init__(self):
        self.video_capture = None

    def start(self):
        self.video_capture = cv2.VideoCapture(0)

    def get_frame(self):
        ret, frame = self.video_capture.read()
        return frame

    def stop(self):
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None

    def is_playing(self):
        return self.video_capture is not None


class VideoPlayback(Playback):
    def __init__(self, path):
        self.video_capture = None
        self.path = path

    def start(self):
        self.video_capture = cv2.VideoCapture(self.path)

    def get_frame(self):
        ret, frame = self.video_capture.read()
        return frame

    def stop(self):
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None

    def is_playing(self):
        return self.video_capture is not None and self.video_capture.isOpened()


class YoutubePlayback(Playback):
    def __init__(self, url):
        self.video_capture = None
        v_pafy = pafy.new(url)
        self.youtube = v_pafy.getbest()

    def start(self):
        self.video_capture = cv2.VideoCapture(self.youtube.url)

    def get_frame(self):
        ret, frame = self.video_capture.read()
        return frame

    def stop(self):
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None

    def is_playing(self):
        return self.video_capture is not None
