#!/usr/bin/env python
#
# This file is part of the Emotions project. The complete source code is
# available at https://github.com/luigivieira/emotions.
#
# Copyright (c) 2016-2017, Luiz Carlos Vieira (http://www.luiz.vieira.nom.br)
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import numpy as np
import dlib
import cv2

from data import FaceData

#=============================================
class FaceDetector:
    """
    Implements the detector of faces (with their landmarks) in images.
    """

    _detector = None
    """
    Instance of the dlib's object used to detect faces in images, shared by all
    instances of this class.
    """

    _predictor = None
    """
    Instance of the dlib's object used to predict the positions of facial
    landmarks in images, shared by all instances of this class.
    """

    #---------------------------------------------
    def detect(self, image, downSampleRatio = None):
        """
        Tries to automatically detect a face in the given image.

        This method uses the face detector/predictor from the dlib package (with
        its default face model) to detect a face region and 68 facial landmarks.
        Even though dlib is able to detect more than one face in the image, for
        the current purposes of the fsdk project only a single face is needed.
        Hence, only the biggest face detected (estimated from the region size)
        is considered.

        Parameters
        ------
        image: numpy.array
            Image data where to search for the face.
        downSampleRatio: float

        Returns
        ------
        result: bool
            Indication on the success or failure of the facial detection.
        face: FaceData
            Instance of the FaceData class with the region and landmarks of the
            detected face, or None if no face was detected.
        """

        #####################
        # Setup the detector
        #####################

        # Initialize the static detector and predictor if this is first use
        if FaceDetector._detector is None or FaceDetector._predictor is None:
            FaceDetector._detector = dlib.get_frontal_face_detector()

            faceModel = os.path.abspath('{}/models/face_model.dat' \
                            .format(os.path.dirname(__file__)))
            FaceDetector._predictor = dlib.shape_predictor(faceModel)

        #####################
        # Performance cues
        #####################

        # If requested, scale down the original image in order to improve
        # performance in the initial face detection
        if downSampleRatio is not None:
            detImage = cv2.resize(image, (0, 0), fx=1.0 / downSampleRatio,
                                                 fy=1.0 / downSampleRatio)
        else:
            detImage = image

        #####################
        # Face detection
        #####################

        # Detect faces in the image
        detectedFaces = FaceDetector._detector(detImage, 1)
        if len(detectedFaces) == 0:
            return False, None

        # No matter how many faces have been found, consider only the first one
        region = detectedFaces[0]

        # If downscaling was requested, scale back the detected region so the
        # landmarks can be proper located on the image in full resolution
        if downSampleRatio is not None:
            region = dlib.rectangle(region.left() * downSampleRatio,
                                    region.top() * downSampleRatio,
                                    region.right() * downSampleRatio,
                                    region.bottom() * downSampleRatio)

        # Fit the shape model over the face region to predict the positions of
        # its facial landmarks
        faceShape = FaceDetector._predictor(image, region)

        #####################
        # Return data
        #####################

        face = FaceData()

        # Update the object data with the predicted landmark positions and
        # their bounding box (with a small margin of 10 pixels)
        face.landmarks = np.array([[p.x, p.y] for p in faceShape.parts()])

        margin = 10
        x, y, w, h = cv2.boundingRect(face.landmarks)
        face.region = (
                       max(x - margin, 0),
                       max(y - margin, 0),
                       min(x + w + margin, image.shape[1] - 1),
                       min(y + h + margin, image.shape[0] - 1)
                      )

        return True, face