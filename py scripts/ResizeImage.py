"""Grasshopper Script"""
import System
import Rhino
import Grasshopper
import cv2

class MyComponent(Grasshopper.Kernel.GH_ScriptInstance):
    def RunScript(self, image, width: int, height: int):
        resize_img = cv2.resize(image, (width, height))
        return resize_img
