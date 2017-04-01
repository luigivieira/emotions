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

import cv2
from collections import OrderedDict
import numpy as np

#=============================================
class FaceData:
    """
    Represents the data of a face detected on an image.
    """

    _jawLine = [i for i in range(17)]
    """
    Indexes of the landmarks at the jaw line.
    """

    _rightEyebrow = [i for i in range(17,22)]
    """
    Indexes of the landmarks at the right eyebrow.
    """

    _leftEyebrow = [i for i in range(22,27)]
    """
    Indexes of the landmarks at the left eyebrow.
    """

    _noseBridge = [i for i in range(27,31)]
    """
    Indexes of the landmarks at the nose bridge.
    """

    _lowerNose = [i for i in range(30,36)]
    """
    Indexes of the landmarks at the lower nose.
    """

    _rightEye = [i for i in range(36,42)]
    """
    Indexes of the landmarks at the right eye.
    """

    _leftEye = [i for i in range(42,48)]
    """
    Indexes of the landmarks at the left eye.
    """

    _outerLip = [i for i in range(48,60)]
    """
    Indexes of the landmarks at the outer lip.
    """

    _innerLip = [i for i in range(60,68)]
    """
    Indexes of the landmarks at the inner lip.
    """

    #---------------------------------------------
    def __init__(self, region = (0, 0, 0, 0),
                 landmarks = [0 for i in range(136)]):
        """
        Class constructor.

        Parameters
        ----------
        region: tuple
            Left, top, right and bottom coordinates of the region where the face
            is located in the image used for detection. The default is all 0's.
        landmarks: list
            List of x, y coordinates of the 68 facial landmarks in the image
            used for detection. The default is all 0's.
        """

        self.region = region
        """
        Region where the face is found in the image used for detection. This is
        a tuple of int values describing the region in terms of the top-left and
        bottom-right coordinates where the face is located.
        """

        self.landmarks = landmarks
        """
        Coordinates of the landmarks on the image. This is a numpy array of
        pair of values describing the x and y positions of each of the 68 facial
        landmarks.
        """

    #---------------------------------------------
    def copy(self):
        """
        Deep copies the data of the face.

        Deep copying means that no mutable attribute (like tuples or lists) in
        the new copy will be shared with this instance. In that way, the two
        copies can be changed independently.

        Returns
        -------
        ret: FaceData
            New instance of the FaceDate class deep copied from this instance.
        """
        return FaceData(self.region, self.landmarks.copy())

    #---------------------------------------------
    def isEmpty(self):
        """
        Check if the FaceData object is empty.

        An empty FaceData object have region and landmarks with all 0's.

        Returns
        ------
        response: bool
            Indication on whether this object is empty.
        """
        return all(v == 0 for v in self.region) or \
               all(vx == 0 and vy == 0 for vx, vy in self.landmarks)

    #---------------------------------------------
    def crop(self, image):
        """
        Crops the given image according to this instance's region and landmarks.

        This function creates a subregion of the original image according to the
        face region coordinates, and also a new instance of FaceDate object with
        the region and landmarks adjusted to the cropped image.

        Parameters
        ----------
        image: numpy.array
            Image that contains the face.

        Returns
        -------
        croppedImage: numpy.array
            Subregion in the original image that contains only the face. This
            image is shared with the original image (i.e. its data is not
            copied, and changes to either the original image or this subimage
            will affect both instances).

        croppedFace: FaceData
            New instance of FaceData with the face region and landmarks adjusted
            to the croppedImage.
        """
        left = self.region[0]
        top = self.region[1]
        right = self.region[2]
        bottom = self.region[3]

        croppedImage = image[top:bottom+1, left:right+1]

        croppedFace = self.copy()
        croppedFace.region = (0, 0, right - left, bottom - top)
        croppedFace.landmarks = [[p[0]-left, p[1]-top] for p in self.landmarks]

        return croppedImage, croppedFace

    #---------------------------------------------
    def draw(self, image, drawRegion = None, drawFaceModel = None):
        """
        Draws the face data over the given image.

        This method draws the facial landmarks (in red) to the image. It can
        also draw the region where the face was detected (in blue) and the face
        model used by dlib to do the prediction (i.e., the connections between
        the landmarks, in magenta). This drawing is useful for visual inspection
        of the data - and it is fun! :)

        Parameters
        ------
        image: numpy.array
            Image data where to draw the face data.
        drawRegion: bool
            Optional value indicating if the region area should also be drawn.
            The default is True.
        drawFaceModel: bool
            Optional value indicating if the face model should also be drawn.
            The default is True.

        Returns
        ------
        drawnImage: numpy.array
            Image data with the original image received plus the face data
            drawn. If this instance of Face is empty (i.e. it has no region
            and no landmarks), the original image is simply returned with
            nothing drawn on it.
        """
        if self.isEmpty():
            raise RuntimeError('Can not draw the contents of an empty '
                               'FaceData object')

        # Check default arguments
        if drawRegion is None:
            drawRegion = True
        if drawFaceModel is None:
            drawFaceModel = True

        # Draw the region if requested
        if drawRegion:
            cv2.rectangle(image, (self.region[0], self.region[1]),
                                 (self.region[2], self.region[3]),
                                 (0, 0, 255), 2)

        # Draw the positions of landmarks
        color = (0, 255, 255)
        for i in range(68):
            cv2.circle(image, tuple(self.landmarks[i]), 1, color, 2)

        # Draw the face model if requested
        if drawFaceModel:
            c = (0, 255, 255)
            p = np.array(self.landmarks)

            cv2.polylines(image, [p[FaceData._jawLine]], False, c, 2)
            cv2.polylines(image, [p[FaceData._leftEyebrow]], False, c, 2)
            cv2.polylines(image, [p[FaceData._rightEyebrow]], False, c, 2)
            cv2.polylines(image, [p[FaceData._noseBridge]], False, c, 2)
            cv2.polylines(image, [p[FaceData._lowerNose]], True, c, 2)
            cv2.polylines(image, [p[FaceData._leftEye]], True, c, 2)
            cv2.polylines(image, [p[FaceData._rightEye]], True, c, 2)
            cv2.polylines(image, [p[FaceData._outerLip]], True, c, 2)
            cv2.polylines(image, [p[FaceData._innerLip]], True, c, 2)

        return image

#=============================================
class GaborData:
    """
    Represents the responses of the Gabor bank to the facial landmarks.
    """

    #---------------------------------------------
    def __init__(self, features = [0.0 for i in range(2176)]):
        """
        Class constructor.

        Parameters
        ----------
        features: list
            Responses of the filtering with the bank of Gabor kernels at each of
            the facial landmarks. The default is all 0's.
        """
        self.features = features
        """
        Responses of the filtering with the bank of Gabor kernels at each of the
        facial landmarks. The Gabor bank used has 32 kernels and there are 68
        landmarks, hence this is a vector of 2176 values (32 x 68).
        """

    #---------------------------------------------
    def copy(self):
        """
        Deep copies the data of this object.

        Deep copying means that no mutable attribute (like tuples or lists) in
        the new copy will be shared with this instance. In that way, the two
        copies can be changed independently.

        Returns
        -------
        ret: GaborData
            New instance of the GaborData class deep copied from this instance.
        """
        return GaborData(self.features.copy())

    #---------------------------------------------
    def isEmpty(self):
        """
        Check if the object is empty.

        Returns
        ------
        response: bool
            Indication on whether this object is empty.
        """
        return all(v == 0 for v in self.features)