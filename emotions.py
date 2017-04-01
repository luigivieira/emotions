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
from collections import OrderedDict
import numpy as np

from gabor import GaborBank
from data import FaceData
from faces import FaceDetector

from sklearn import svm
from sklearn.externals import joblib

#=============================================
class InvalidModelException(Exception):
    '''
    Exception indicating that the detection model could not be loaded (or didn't
    exist).
    '''
    pass

#=============================================
class EmotionsDetector:
    """
    Implements the detector of prototypic emotions on face images.
    """

    #---------------------------------------------
    def __init__(self):
        """
        Class constructor.
        """

        self._clf = svm.SVC(kernel='rbf', gamma=0.001, C=10,
                            decision_function_shape='ovr',
                            probability=True, class_weight='balanced')
        """
        Support Vector Machine with used as the model for the detection of the
        prototypic emotions. For details on the selection of the kernel and
        parameters, refer to the PhD thesis of the author of this code.
        """

        self._emotions = OrderedDict([
                             (0, 'neutral'), (1, 'happiness'), (2, 'sadness'),
                             (3, 'anger'), (4, 'fear'),  (5, 'surprise'),
                             (6, 'disgust')
                         ])
        """
        Class and labels of the prototypic emotions detected by this model.
        """

        modulePath = os.path.dirname(__file__)
        self._modelFile = os.path.abspath('{}/models/emotions_model.dat' \
                            .format(modulePath))
        """
        Name of the file used to persist the model in the disk.
        """

        # Load the model from the disk, if its file exists
        if not os.path.isfile(self._modelFile):
            raise InvalidModelException('Could not find model file: {}' \
                    .format(self._modelFile))

        if not self.load():
            raise InvalidModelException('Could not load model from file: {}' \
                    .format(self._modelFile))

    #---------------------------------------------
    def load(self):
        """
        Loads the SVM model from the disk.

        Returns
        -------
        ret: bool
            Indication on if the loading was succeeded or not.
        """

        try:
            clf = joblib.load(self._modelFile)
        except:
            return False

        self._clf = clf
        return True

    #---------------------------------------------
    def _relevantFeatures(self, gaborResponses, facialLandmarks):
        """
        Get the features that are relevant for the detection of emotions
        from the matrix of responses to the bank of Gabor kernels.

        The feature vector returned by this method can be used for training and
        predicting, using a linear SVM.

        Parameters
        ----------
        gaborResponses: numpy.array
            Matrix of responses to the bank of Gabor kernels applied to the face
            region of an image. The first dimension of this matrix has size 32,
            one for each kernel in the bank. The other two dimensions are in the
            same size as the original image used for their extraction.

        facialLandmarks: numpy.array
            Bidimensional matrix with the coordinates of each facial landmark
            detected in the face image from where the responses were obtained.

        Returns
        -------
        featureVector: list
            A list with the responses of the 32 kernels at each of the
            face landmarks.
        """

        # Get the 32 responses at the positions of all the face landmarks
        points = np.array(facialLandmarks)

        # Try to get the responses for all points. If an exception is caught,
        # it is because some landmarks are out of the image area (i.e. the face
        # is partially occluded, but it was still possible to detect). In this
        # case, assume 0.0 for the responses of the landmarks outside the image
        # area.
        try:
            responses = gaborResponses[:, points[:, 1], points[:, 0]]
        except:
            w = gaborResponses.shape[2]
            h = gaborResponses.shape[1]

            responses = np.zeros((32, 68), dtype=float)
            for i in range(len(points)):
                x = points[i][0]
                y = points[i][1]
                if x < w and y < h:
                    responses[:, i] = gaborResponses[:, y, x]
                else:
                    responses[:, i] = 0.0

        # Reshape the bi-dimensional matrix to a single dimension
        featureVector = responses.reshape(-1).tolist()

        return featureVector

    #---------------------------------------------
    def detect(self, face, gaborResponses):
        """
        Detects the emotions based on the given features.

        Parameters
        ----------
        face: FaceData
            Instance of the FaceData object with the facial landmarks detected
            on the facial image.
        gaborResponses: numpy.array
            Matrix of responses to the bank of Gabor kernels applied to the face
            region of an image. The first dimension of this matrix has size 32,
            one for each kernel in the bank. The other two dimensions are in the
            same size as the original image used for their extraction.

        Returns
        -------
        probabilities: OrderedDict
            The probabilities of each of the prototypic emotion, in format:
            {'anger': value, 'contempt': value, [...]}
        """

        # Filter only the responses at the facial landmarks
        features = self._relevantFeatures(gaborResponses, face.landmarks)

        # Return the prediction based on the given features
        return self.predict(features)

    #---------------------------------------------
    def predict(self, features):
        """
        Predicts the emotions on the given features vector.

        Parameters
        ----------
        features: list
            List of responses of the kernels at each of the face landmarks.

        Returns
        -------
        probabilities: OrderedDict
            The probabilities of each of the prototypic emotion, in format:
            {'anger': value, 'contempt': value, [...]}
        """

        # Predict the emotion probabilities on the given features
        probas = self._clf.predict_proba([features])[0]

        # Build a dictionary with the probabilities and emotion labels
        ret = OrderedDict()
        for i in range(len(self._emotions)):
            label = self._emotions[i]
            ret[label] = probas[i]

        return ret