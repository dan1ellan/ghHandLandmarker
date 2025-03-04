"""Grasshopper Script"""
import System
import Rhino
import Grasshopper
import Grasshopper.Kernel as gh
import cv2
import os
import numpy as np

class MyComponent(Grasshopper.Kernel.GH_ScriptInstance):
    def RunScript(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[1]

        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if (contours == None):
            ghenv.Component.AddRuntimeMessage(gh.GH_RuntimeMessageLevel.Error, "find contour failed.")
            return

        c = max(contours[0], key=cv2.contourArea)
        contour_length = cv2.arcLength(c, True)

        approx_contours = None
        for e in np.linspace(0.01, 0.1, 10):
            approx = cv2.approxPolyDP(c, e * contour_length, True)
            if len(approx) <= 4:
                break
            approx_contours = approx

        if (approx_contours is None):
            ghenv.Component.AddRuntimeMessage(gh.GH_RuntimeMessageLevel.Error, "failed to approximate contour.")
            return

        hull = cv2.convexHull(approx_contours)

        if (len(hull) != 4):
            ghenv.Component.AddRuntimeMessage(gh.GH_RuntimeMessageLevel.Warning, "number of corners found is not 4.")

        sort_hull_x_desc = hull[np.argsort(hull[:,0,0])[::-1]]
        top_two_max_x = sort_hull_x_desc[:2]
        first_pt = top_two_max_x[np.argmin(top_two_max_x[:,0,1])]
        first_pt_index = np.where(hull == first_pt)[0][0]
        sorted_hull = np.roll(hull, -first_pt_index, axis = 0)

        # corners = []
        # for i in range(len(hull)):
        #     corners.append(Rhino.Geometry.Point3d(float(hull[i][0][0]), float(hull[i][0][1]), 0))

        return sorted_hull
