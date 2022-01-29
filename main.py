import cv2
import numpy as np
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

from tracking import HandDetector


def main():
    modules = init_modules()
    track_volume(*modules)


def init_modules():
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    min_vol, max_vol, _ = volume.GetVolumeRange()
    detector = HandDetector(detect_confidence=0.7)
    return volume, min_vol, max_vol, detector


def track_volume(volume, min_vol, max_vol, detector):
    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()

        img = detector.draw_landmarks(img)
        landmarks = detector.hands_position(img)

        if landmarks:
            x1, y1 = landmarks[4][1], landmarks[4][2]
            x2, y2 = landmarks[8][1], landmarks[8][2]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

            length = math.hypot(x2 - x1, y2 - y1)
            cur_vol = np.interp(length, [35, 200], [min_vol, max_vol])
            volume.SetMasterVolumeLevel(cur_vol, None)

            if length < 35 or length > 200:
                cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

        cv2.imshow('Video', img)
        key = cv2.waitKey(1)

        if key == 113 or key == 233:
            break


if __name__ == '__main__':
    main()
