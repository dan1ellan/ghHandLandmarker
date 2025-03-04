"""Grasshopper Script"""
import System
import Rhino
import Grasshopper
import Grasshopper.Kernel as gh
import cv2
import numpy as np
import math


class MyComponent(Grasshopper.Kernel.GH_ScriptInstance):
    def RunScript(self, image, corners):
        if len(corners) != 4:
            ghenv.Component.AddRuntimeMessage(gh.GH_RuntimeMessageLevel.Error, "number of corners must be 4.")
            return

        input_corners = np.float32(corners)

        width, height = get_width_height(corners)

        output_corners = np.float32([[width, 0], [width, height], [0, height], [0, 0]])

        M = cv2.getPerspectiveTransform(input_corners, output_corners)
        warp_image = cv2.warpPerspective(image, M, (int(width), int(height)))

        return warp_image


def get_width_height(corners):
    p0 = corners[0]
    p1 = corners[1]
    p2 = corners[2]
    p3 = corners[3]

    d01 = math.sqrt((p1[0][0] - p0[0][0]) ** 2 + (p1[0][1] - p0[0][1]) ** 2)
    d12 = math.sqrt((p2[0][0] - p1[0][0]) ** 2 + (p2[0][1] - p1[0][1]) ** 2)
    d23 = math.sqrt((p3[0][0] - p2[0][0]) ** 2 + (p3[0][1] - p2[0][1]) ** 2)
    d30 = math.sqrt((p0[0][0] - p3[0][0]) ** 2 + (p0[0][1] - p3[0][1]) ** 2)

    width = (d30 + d12) / 2
    height = (d01 + d23) / 2

    print(width)
    print(height)
    return width, height
