"""Grasshopper Script Instance"""
import System
import Rhino
import Grasshopper
import cv2
import numpy as np
import clr
import os
from System.IO import MemoryStream
from System.Drawing import Bitmap


class MyComponent(Grasshopper.Kernel.GH_ScriptInstance):
    def RunScript(self, image):
        image_enc = cv2.imencode(".jpg", image)[1]
        image_bytes = image_enc.tobytes()
        ms = MemoryStream(image_bytes, True)
        bitmap = Bitmap(ms)
        return bitmap
        
