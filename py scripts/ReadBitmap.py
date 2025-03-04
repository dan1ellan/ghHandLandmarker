"""Grasshopper Script"""
import System
import Rhino
import Grasshopper
import os
import cv2 
import numpy as np
from System.Drawing import Bitmap
from System.IO import MemoryStream
from System.Drawing.Imaging import ImageFormat

DIRECTORY = os.path.dirname(ghdoc.Path)

class MyComponent(Grasshopper.Kernel.GH_ScriptInstance):
    def RunScript(self, bitmap):
        ms = MemoryStream()
        bitmap.Save(ms, ImageFormat.Jpeg)
        image_array = np.frombuffer(ms.ToArray(), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR);
        return image
