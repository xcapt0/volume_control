import cv2
import mediapipe as mp


class HandDetector():
    def __init__(self, mode=False, max_hands=2, complexity=1, detect_confidence=0.5, track_confidence=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.complexity = complexity
        self.detect_confidence = detect_confidence
        self.track_confidence = track_confidence
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode,
                                         self.max_hands,
                                         self.complexity,
                                         self.detect_confidence,
                                         self.track_confidence)
        self.mp_draw = mp.solutions.drawing_utils

    def draw_landmarks(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = self.hands.process(img_rgb)

        self.multiple_hands = res.multi_hand_landmarks
        if self.multiple_hands:
            for hand in self.multiple_hands:
                self.mp_draw.draw_landmarks(img, hand, self.mp_hands.HAND_CONNECTIONS)

        return img

    def hands_position(self, img, hand_id=0):
        landmarks = []

        if self.multiple_hands:
            hand = self.multiple_hands[hand_id]
            for id, lm in enumerate(hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks.append([id, cx, cy])

        return landmarks
