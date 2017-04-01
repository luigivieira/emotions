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

import numpy as np
from skimage.filters import gabor_kernel
import cv2

#=============================================
class KernelParams:
    """
    A simple class to represent the parameters of a given Gabor kernel.
    """

    #---------------------------------------------
    def __init__(self, wavelength, orientation):
        """
        Class constructor. Define the parameters of a Gabor kernel.

        Parameters
        ----------
        wavelength: float
            Wavelength (in pixels) of a Gabor kernel.
        orientation: float
            Orientations (in radians) of a Gabor kernel.
        """

        self.wavelength = wavelength
        """Wavelength (in pixels) of a Gabor kernel."""

        self.orientation = orientation
        """Orientation (in radians) of a Gabor kernel."""

    #---------------------------------------------
    def __hash__(self):
        """
        Generates a hash value for this object instance.

        Returns
        ----------
        hash: int
            Hash value of this object.
        """
        return hash((self.wavelength, self.orientation))

    #---------------------------------------------
    def __eq__(self, other):
        """
        Verifies if this object instance is equal to another.

        This method is the implementation of the == operator.

        Parameters
        ----------
        other: KernelParams
            Other instance to compare with this one.

        Returns
        ----------
        eq: bool
            True if this and the other instances have the same parameters, or
            False otherwise.
        """
        return (self.wavelength, self.orientation) == \
               (other.wavelength, other.orientation)

    #---------------------------------------------
    def __ne__(self, other):
        """
        Verifies if this object instance is different than another.

        This method is the implementation of the != operator.

        Parameters
        ----------
        other: KernelParams
            Other instance to compare with this one.

        Returns
        ----------
        neq: bool
            True if this and the other instances have different parameters, or
            False otherwise.
        """
        return not(self == other)

#=============================================
class GaborBank:
    """
    Represents a bank of gabor kernels.
    """

    #---------------------------------------------
    def __init__(self, w = [4, 7, 10, 13],
                       o = [i for i in np.arange(0, np.pi, np.pi / 8)]):
        """
        Class constructor. Create a bank of Gabor kernels with a predefined set
        of wavelengths and orientations.

        The bank is composed of one kernel for each combination of wavelength x
        orientation. For the rationale regarding the choice of parameters, refer
        to the PhD thesis of the author of this code.
        """

        self._wavelengths = w
        """
        List of wavelengths (in pixels) used to create the bank of Gabor
        kernels.
        """

        self._orientations = o
        """
        List of orientations (in radians) used to create the bank of Gabor
        kernels.
        """

        self._kernels = {}
        """Dictionary holding the Gabor kernels in the bank."""

        # Create one kernel for each combination of wavelength x orientation
        for wavelength in self._wavelengths:
            for orientation in self._orientations:
                # Convert wavelength to spatial frequency (scikit-image's
                # interface expects spatial frequency, even though the original
                # equation uses wavelengths - see https://en.wikipedia.org/wiki/
                # Gabor_filter/)
                frequency = 1 / wavelength

                # Create and save the kernel
                kernel = gabor_kernel(frequency, orientation)
                par = KernelParams(wavelength, orientation)
                self._kernels[par] = kernel

    #---------------------------------------------
    def filter(self, image):
        """
        Filter the given image with the Gabor kernels in this bank.

        Parameters
        ----------
        image: numpy.array
            Image to be filtered.

        Returns
        -------
        responses: numpy.array
            List of the responses of the filtering with the Gabor kernels. The
            responses are the magnitude of both the real and imaginary parts of
            the convolution with each kernel, hence this list dimensions are the
            same of the image, plus another dimension for the 32 responses (one
            for each kernel in the bank, since there are 4 wavelengths and 8
            orientations).
        """

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        responses = []
        for wavelength in self._wavelengths:
            for orientation in self._orientations:

                # Get the kernel
                frequency = 1 / wavelength
                par = KernelParams(wavelength, orientation)
                kernel = self._kernels[par]

                # Filter with both real and imaginary parts
                real = cv2.filter2D(image, cv2.CV_32F, kernel.real)
                imag = cv2.filter2D(image, cv2.CV_32F, kernel.imag)

                # The response is the magnitude of the real and imaginary
                # responses to the filters, normalized to [-1, 1]
                mag = cv2.magnitude(real, imag)
                cv2.normalize(mag, mag, -1, 1, cv2.NORM_MINMAX)

                responses.append(mag)

        return np.array(responses)