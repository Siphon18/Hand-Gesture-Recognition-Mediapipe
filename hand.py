import argparse
import cv2
import mediapipe as mp
import time
import numpy as np

from utils.utils import check_hand_direction, find_boundary_lm
from utils.utils import calculate_angle, display_hand_info

CAM_W = 1280
CAM_H = 720
TEXT_COLOR = (243,236,27)
LM_COLOR = (102,255,255)
LINE_COLOR = (51,51,51)


class HandDetector:
    def __init__(self, static_image_mode=False, max_num_hands=2,
                 min_detection_confidence=0.8, min_tracking_confidence=0.5):
        """
        Initialize the HandDetector with the specified parameters.
        """
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        # MediaPipe Hands module
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        # Initialize MediaPipe Hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
    
    def detect_hands(self, img):
        """
        Detect hands in the image and return the landmarks, handedness, and other features.
        """
        self.decoded_hands = None
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        
        if self.results.multi_hand_landmarks:
            h, w, _ = img.shape
            num_hands = len(self.results.multi_hand_landmarks)
            self.decoded_hands = [None] * num_hands

            # Iterate through each detected hand
            for i in range(num_hands):
                self.decoded_hands[i] = dict()
                lm_list = list()
                handedness = self.results.multi_handedness[i]
                hand_landmarks = self.results.multi_hand_landmarks[i]
                wrist_z = hand_landmarks.landmark[0].z  # Wrist Z coordinate for depth

                # Process landmarks for each hand
                for lm in hand_landmarks.landmark:
                    cx = int(lm.x * w)  # X coordinate
                    cy = int(lm.y * h)  # Y coordinate
                    cz = int((lm.z - wrist_z) * w)  # Z coordinate (relative to wrist depth)
                    lm_list.append([cx, cy, cz])

                # Extract handedness and other information
                label = handedness.classification[0].label.lower()
                lm_array = np.array(lm_list)
                direction, facing = check_hand_direction(lm_array, label)
                boundary = find_boundary_lm(lm_array)
                wrist_angle_joints = lm_array[[5, 0, 17]]  # Wrist angle calculation
                wrist_angle = calculate_angle(wrist_angle_joints)

                # Store hand data in dictionary
                self.decoded_hands[i]['label'] = label
                self.decoded_hands[i]['landmarks'] = lm_array
                self.decoded_hands[i]['wrist_angle'] = wrist_angle
                self.decoded_hands[i]['direction'] = direction
                self.decoded_hands[i]['facing'] = facing
                self.decoded_hands[i]['boundary'] = boundary
        
        return self.decoded_hands
    
    def draw_landmarks(self, img):
        """
        Draw hand landmarks on the image.
        """
        if self.results.multi_hand_landmarks:
            for landmarks in self.results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    img, landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=LM_COLOR, thickness=3),
                    self.mp_drawing.DrawingSpec(color=LINE_COLOR, thickness=2)
                )


def main(max_hands=2):
    """
    Main function to capture the video feed and process the frames for hand detection.
    """
    cap = cv2.VideoCapture(0)
    cap.set(3, CAM_W)
    cap.set(4, CAM_H)
    detector = HandDetector(max_num_hands=max_hands)
    ptime = 0
    ctime = 0

    while True:
        # Read the frame
        _, img = cap.read()
        img = cv2.flip(img, 1)

        # Detect hands and draw landmarks
        detector.detect_hands(img)
        detector.draw_landmarks(img)

        # Display hand information if hands are detected
        if detector.decoded_hands:
            for hand in detector.decoded_hands:
                display_hand_info(img, hand)

        # Calculate FPS
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime

        # Display FPS on the frame
        cv2.putText(img, f'FPS: {int(fps)}', (50, 50), 0, 0.8, TEXT_COLOR, 2, lineType=cv2.LINE_AA)

        # Show the frame
        cv2.imshow('Hand detection', img)

        # Break loop if 'q' is pressed
        key = cv2.waitKey(1)
        if key == ord('q'):
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    # Argument parser for command-line options
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_hands', type=int, default=2, help='max number of hands (default: 2)')
    opt = parser.parse_args()

    main(**vars(opt))
