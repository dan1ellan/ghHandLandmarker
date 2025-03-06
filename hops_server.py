# Mediapipe hand landmarker reference: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/index#models
# Hops server reference: https://github.com/mcneel/compute.rhino3d/tree/8.x/src/ghhops-server-py
import ghhops_server as hs
import rhino3dm
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import os
import math
import base64

# register hops app as middleware
hops = hs.Hops()

DIRECTORY = os.path.dirname(os.path.realpath(__file__))

def encoded_str_to_image(encoded_str: str):
    buffer = base64.b64decode(encoded_str)
    nparr = np.frombuffer(buffer, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def image_to_encoded_str(image):
    buffer = cv2.imencode('.jpg', image)[1]
    encoded_str = base64.b64encode(buffer).decode()
    return encoded_str

@hops.component(
    "/readimage",
    name="ReadImage",
    description="Read image from file and encode it to string",
    inputs=[
        hs.HopsString("ImagePath", "Path", "Image to read")
    ],
    outputs=[
        hs.HopsString("EncodedImage", "String", "Encoded image string")
    ]
)
def read_image(img_path):
    image = cv2.imread(img_path)
    encoded_str = image_to_encoded_str(image)
    return encoded_str

@hops.component(
    "/writeimage",
    name="WriteImage",
    description="Write encoded image string to file",
    inputs=[
        hs.HopsString("EncodedImage", "String", "Encoded image string")
    ],
    outputs=[
        hs.HopsString("ImagePath", "Path", "Image path")
    ]
)
def write_image(encoded_str):
    image = encoded_str_to_image(encoded_str)
    path = os.path.join(DIRECTORY, "output.jpg")
    cv2.imwrite(path, image)
    return path


@hops.component(
    "/resizeimage",
    name="ResizeImage",
    description="Resizes image to given width and height",
    inputs=[
        hs.HopsString("EncodedImage", "String", "Encoded image string"),
        hs.HopsInteger("Width", "Width", "Width of image"),
        hs.HopsInteger("Height", "Height", "Height of image")
    ],
    outputs=[
        hs.HopsString("ResizedEncodedImage", "String", "Resized encoded image string"),
    ]
)
def resize_image(encoded_str, width, height):
    image = encoded_str_to_image(encoded_str)
    image = cv2.resize(image, (width, height))
    return image_to_encoded_str(image)



@hops.component(
    "/handlandmark",
    name="HandLandmark",
    description="Get landmark points from photo of one hand",
    inputs=[
        hs.HopsString("EncodedImage", "String", "Encoded image string"),
    ],
    outputs=[
        hs.HopsBoolean("Handedness", "Bool", "True if right hand, False if left hand"),
        hs.HopsPoint("Points", "Points", "Landmark points"),
        hs.HopsString("AnnotatedEncodedImage", "String", "Annotated encoded image string")
    ]
)
def hand_landmarker(encoded_str):
    with solutions.hands.Hands(static_image_mode=True, max_num_hands=1) as detector:
        img = encoded_str_to_image(encoded_str)
        img = cv2.flip(img, 1)
        img_cvt = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detection_result = detector.process(img_cvt)

        hand_landmarks = detection_result.multi_hand_landmarks[0]
        handedness = detection_result.multi_handedness[0].classification[0].label

        points = []
        h, w, _ = img.shape
        for landmark in hand_landmarks.landmark:
            points.append(
                rhino3dm.Point3d((1-landmark.x) * w, (1-landmark.y) * h, 0)
            )
        img = cv2.flip(img, 1)

        annotated_image = annotate_landmarks(img, hand_landmarks, handedness)

        return handedness == "Right", points, annotated_image


def annotate_landmarks(image, hand_landmarks, handedness):
    annotated_image = np.copy(image)

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

    return image_to_encoded_str(annotated_image)


@hops.component(
    "/cornerdetection",
    name="CornerDetection",
    description="Get corner points from photo",
    inputs=[
        hs.HopsString("EncodedImage", "String", "Encoded image string"),
    ],
    outputs=[
        hs.HopsPoint("Corners", "Points", "Corners")
    ]
)
def detect_corners(encoded_str):
    image = encoded_str_to_image(encoded_str)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[1]

    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if (contours is None):
        raise TypeError("failed to find contours.")

    c = max(contours[0], key=cv2.contourArea)
    contour_length = cv2.arcLength(c, True)

    approx_contours = None
    for e in np.linspace(0.01, 0.1, 10):
        approx = cv2.approxPolyDP(c, e * contour_length, True)
        if len(approx) <= 4:
            break
        approx_contours = approx

    if (approx_contours is None):
        raise TypeError("failed to find approx contours.")

    hull = cv2.convexHull(approx_contours)

    if (len(hull) != 4):
        raise ValueError("failed to find approx contours.")

    sort_hull_x_desc = hull[np.argsort(hull[:,0,0])[::-1]]
    top_two_max_x = sort_hull_x_desc[:2]
    first_pt = top_two_max_x[np.argmin(top_two_max_x[:,0,1])]
    first_pt_index = np.where(hull == first_pt)[0][0]
    sorted_hull = np.roll(hull, -first_pt_index, axis = 0)

    corners = []
    for i in range(len(hull)):
        corners.append(rhino3dm.Point3d(float(sorted_hull[i][0][0]), float(sorted_hull[i][0][1]), 0))

    return corners


@hops.component(
    "/warpperspective",
    name="WarpPerspective",
    description="Warp perspective of image",
    inputs=[
        hs.HopsString("EncodedImage", "String", "Encoded image string"),
        hs.HopsPoint("Corners", "Points", "Corners", access = hs.HopsParamAccess.LIST),
    ],
    outputs=[
        hs.HopsString("WarpedEncodedImage", "String", "Warped encoded image string")
    ]
)
def warp_perspective(encoded_str, corners: list[rhino3dm.Point3d]):
    if len(corners) != 4:
        raise ValueError("Corners must be 4 points")

    image = encoded_str_to_image(encoded_str)
    input_corners = np.float32([[corners[0].X, corners[0].Y], 
                                [corners[1].X, corners[1].Y], 
                                [corners[2].X, corners[2].Y], 
                                [corners[3].X, corners[3].Y]])

    width, height = get_width_height(corners)

    output_corners = np.float32([[width, 0], [width, height], [0, height], [0, 0]])

    M = cv2.getPerspectiveTransform(input_corners, output_corners)
    warp_image = cv2.warpPerspective(image, M, (int(width), int(height)))

    return image_to_encoded_str(warp_image)


def get_width_height(corners: list[rhino3dm.Point3d]):
    p0 = corners[0]
    p1 = corners[1]
    p2 = corners[2]
    p3 = corners[3]

    d01 = math.sqrt((p1.X - p0.X) ** 2 + (p1.Y - p0.Y) ** 2)
    d12 = math.sqrt((p2.X - p1.X) ** 2 + (p2.Y - p1.Y) ** 2)
    d23 = math.sqrt((p3.X - p2.X) ** 2 + (p3.Y - p2.Y) ** 2)
    d30 = math.sqrt((p0.X - p3.X) ** 2 + (p0.Y - p3.Y) ** 2)

    width = (d30 + d12) / 2
    height = (d01 + d23) / 2

    return width, height


if __name__ == "__main__":
    hops.start()
