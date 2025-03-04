"""Grasshopper Script"""
import System
import Rhino
import Grasshopper
import cv2

class MyComponent(Grasshopper.Kernel.GH_ScriptInstance):
    def RunScript(self, image, max_dim: int):
        w, h = image.shape[1], image.shape[0]
        scale = max_dim / max(w, h)
        image = cv2.resize(image, (int(w * scale), int(h * scale))) 
        return image
