"""Grasshopper Script"""
# Mediapipe hand landmarker reference: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/index#models
# requirements: mediapipe
import System
import Rhino
import Grasshopper
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import os


DIRECTORY = os.path.dirname(ghdoc.Path)


class MyComponent(Grasshopper.Kernel.GH_ScriptInstance):
    def RunScript(self, img):
        return hand_landmarks(img)


def hand_landmarks(img):
    with solutions.hands.Hands(static_image_mode=True, max_num_hands=1) as detector:
        img = cv2.flip(img, 1)
        img_cvt = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        detection_result = detector.process(img_cvt)

        hand_landmarks = detection_result.multi_hand_landmarks[0]
        handedness = detection_result.multi_handedness[0].classification[0].label

        points = []
        h, w, _ = img.shape
        for landmark in hand_landmarks.landmark:
            points.append(
                Rhino.Geometry.Point3d((1-landmark.x) * w, (1-landmark.y) * h, 0)
            )
        img = cv2.flip(img, 1)

        annotated_image = annotate_landmarks(img, hand_landmarks, handedness)

        return handedness == "Right", points, annotated_image


def annotate_landmarks(rgb_image, hand_landmarks, handedness) -> None:
    annotated_image = np.copy(rgb_image)

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend(
        [
            landmark_pb2.NormalizedLandmark(x=1-landmark.x, y=landmark.y, z=landmark.z)
            for landmark in hand_landmarks.landmark
        ]
    )

    solutions.drawing_utils.draw_landmarks(
        annotated_image,
        hand_landmarks_proto,
        solutions.hands.HAND_CONNECTIONS,
        solutions.drawing_styles.get_default_hand_landmarks_style(),
        solutions.drawing_styles.get_default_hand_connections_style(),
    )

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks.landmark]
    y_coordinates = [landmark.y for landmark in hand_landmarks.landmark]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - 10

    # Draw handedness (left or right hand) on the image.
    cv2.putText(
        annotated_image,
        f"{handedness}",
        (text_x, text_y),
        cv2.FONT_HERSHEY_DUPLEX,
        1,
        (88, 205, 54),
        1,
        cv2.LINE_AA,
    )
    return annotated_image
