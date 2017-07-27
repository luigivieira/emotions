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

import sys
import argparse
import cv2
import numpy as np
from collections import OrderedDict
from datetime import datetime, timedelta

from faces import FaceDetector
from data import FaceData
from gabor import GaborBank
from emotions import EmotionsDetector

#---------------------------------------------
class VideoData:
    """
    Helper class to present the detected face region, landmarks and emotions.
    """

    #-----------------------------------------
    def __init__(self):
        """
        Class constructor.
        """

        self._faceDet = FaceDetector()
        '''
        The instance of the face detector.
        '''

        self._bank = GaborBank()
        '''
        The instance of the bank of Gabor filters.
        '''

        self._emotionsDet = EmotionsDetector()
        '''
        The instance of the emotions detector.
        '''

        self._face = FaceData()
        '''
        Data of the last face detected.
        '''

        self._emotions = OrderedDict()
        '''
        Data of the last emotions detected.
        '''

    #-----------------------------------------
    def detect(self, frame):
        """
        Detects a face and the prototypic emotions on the given frame image.

        Parameters
        ----------
        frame: numpy.ndarray
            Image where to perform the detections from.

        Returns
        -------
        ret: bool
            Indication of success or failure.
        """

        ret, face = self._faceDet.detect(frame)
        if ret:
            self._face = face

            # Crop just the face region
            frame, face = face.crop(frame)

            # Filter it with the Gabor bank
            responses = self._bank.filter(frame)

            # Detect the prototypic emotions based on the filter responses
            self._emotions = self._emotionsDet.detect(face, responses)

            return True
        else:
            self._face = None
            return False

    #-----------------------------------------
    def draw(self, frame):
        """
        Draws the detected data of the given frame image.

        Parameters
        ----------
        frame: numpy.ndarray
            Image where to draw the information to.
        """
        # Font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thick = 1
        glow = 3 * thick

        # Color settings
        black = (0, 0, 0)
        white = (255, 255, 255)
        yellow = (0, 255, 255)
        red = (0, 0, 255)

        empty = True

        # Plot the face landmarks and face distance
        x = 5
        y = 0
        w = int(frame.shape[1]* 0.2)
        try:
            face = self._face
            empty = face.isEmpty()
            face.draw(frame)
        except:
            pass

        # Plot the emotion probabilities
        try:
            emotions = self._emotions
            if empty:
                labels = []
                values = []
            else:
                labels = list(emotions.keys())
                values = list(emotions.values())
                bigger = labels[values.index(max(values))]

                # Draw the header
                text = 'emotions'
                size, _ = cv2.getTextSize(text, font, scale, thick)
                y += size[1] + 20

                cv2.putText(frame, text, (x, y), font, scale, black, glow)
                cv2.putText(frame, text, (x, y), font, scale, yellow, thick)

                y += 5
                cv2.line(frame, (x,y), (x+w,y), black, 1)

            size, _ = cv2.getTextSize('happiness', font, scale, thick)
            t = size[0] + 20
            w = 150
            h = size[1]
            for l, v in zip(labels, values):
                lab = '{}:'.format(l)
                val = '{:.2f}'.format(v)
                size, _ = cv2.getTextSize(l, font, scale, thick)

                # Set a red color for the emotion with bigger probability
                color = red if l == bigger else yellow

                y += size[1] + 15

                p1 = (x+t, y-size[1]-5)
                p2 = (x+t+w, y-size[1]+h+5)
                cv2.rectangle(frame, p1, p2, black, 1)

                # Draw the filled rectangle proportional to the probability
                p2 = (p1[0] + int((p2[0] - p1[0]) * v), p2[1])
                cv2.rectangle(frame, p1, p2, color, -1)
                cv2.rectangle(frame, p1, p2, black, 1)

                # Draw the emotion label
                cv2.putText(frame, lab, (x, y), font, scale, black, glow)
                cv2.putText(frame, lab, (x, y), font, scale, color, thick)

                # Draw the value of the emotion probability
                cv2.putText(frame, val, (x+t+5, y), font, scale, black, glow)
                cv2.putText(frame, val, (x+t+5, y), font, scale, white, thick)
        except Exception as e:
            print(e)
            pass

#---------------------------------------------
def main(argv):
    """
    Main entry of this script.

    Parameters
    ------
    argv: list of str
        Arguments received from the command line.
    """

    # Parse the command line
    args = parseCommandLine(argv)

    # Loads the video or starts the webcam
    if args.source == 'cam':
        video = cv2.VideoCapture(args.id)
        if not video.isOpened():
            print('Error opening webcam of id {}'.format(args.id))
            sys.exit(-1)

        fps = 0
        frameCount = 0
        sourceName = 'Webcam #{}'.format(args.id)
    else:
        video = cv2.VideoCapture(args.file)
        if not video.isOpened():
            print('Error opening video file {}'.format(args.file))
            sys.exit(-1)

        fps = int(video.get(cv2.CAP_PROP_FPS))
        frameCount = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        sourceName = args.file

    # Force HD resolution (if the video was not recorded in this resolution or
    # if the camera does not support it, the frames will be stretched to fit it)
    # The intention is just to standardize the input (and make the help window
    # work as intended)
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280);
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720);

    # Create the helper class
    data = VideoData()

    # Text settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1
    thick = 1
    glow = 3 * thick

    # Color settings
    color = (255, 255, 255)

    paused = False
    frameNum = 0

    # Process the video input
    while True:

        if not paused:
            start = datetime.now()

        ret, img = video.read()
        if ret:
            frame = img.copy()
        else:
            paused = True

        drawInfo(frame, frameNum, frameCount, paused, fps, args.source)

        data.detect(frame)
        data.draw(frame)

        cv2.imshow(sourceName, frame)

        if paused:
            key = cv2.waitKey(0)
        else:
            end = datetime.now()
            delta = (end - start)
            if fps != 0:
                delay = int(max(1, ((1 / fps) - delta.total_seconds()) * 1000))
            else:
                delay = 1

            key = cv2.waitKey(delay)

        if key == ord('q') or key == ord('Q') or key == 27:
            break
        elif key == ord('p') or key == ord('P'):
            paused = not paused
        elif args.source == 'video' and (key == ord('r') or key == ord('R')):
            frameNum = 0
            video.set(cv2.CAP_PROP_POS_FRAMES, frameNum)
        elif args.source == 'video' and paused and key == 2424832: # Left key
            frameNum -= 1
            if frameNum < 0:
                frameNum = 0
            video.set(cv2.CAP_PROP_POS_FRAMES, frameNum)
        elif args.source == 'video' and paused and key == 2555904: # Right key
            frameNum += 1
            if frameNum >= frameCount:
                frameNum = frameCount - 1
        elif args.source == 'video' and key == 2162688: # Pageup key
            frameNum -= (fps * 10)
            if frameNum < 0:
                frameNum = 0
            video.set(cv2.CAP_PROP_POS_FRAMES, frameNum)
        elif args.source == 'video' and key == 2228224: # Pagedown key
            frameNum += (fps * 10)
            if frameNum >= frameCount:
                frameNum = frameCount - 1
            video.set(cv2.CAP_PROP_POS_FRAMES, frameNum)
        elif key == 7340032: # F1
            showHelp(sourceName, frame.shape)

        if not paused:
            frameNum += 1

    video.release()
    cv2.destroyAllWindows()

#---------------------------------------------
def drawInfo(frame, frameNum, frameCount, paused, fps, source):
    """
    Draws text info related to the given frame number into the frame image.

    Parameters
    ----------
    image: numpy.ndarray
        Image data where to draw the text info.
    frameNum: int
        Number of the frame of which to drawn the text info.
    frameCount: int
        Number total of frames in the video.
    paused: bool
        Indication if the video is paused or not.
    fps: int
        Frame rate (in frames per second) of the video for time calculation.
    source: str
        Source of the input images (either "video" or "cam").
    """

    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thick = 1
    glow = 3 * thick

    # Color settings
    black = (0, 0, 0)
    yellow = (0, 255, 255)

    # Print the current frame number and timestamp
    if source == 'video':
        text = 'Frame: {:d}/{:d} {}'.format(frameNum, frameCount - 1,
                                            '(paused)' if paused else '')
    else:
        text = 'Frame: {:d} {}'.format(frameNum, '(paused)' if paused else '')
    size, _ = cv2.getTextSize(text, font, scale, thick)
    x = 5
    y = frame.shape[0] - 2 * size[1]
    cv2.putText(frame, text, (x, y), font, scale, black, glow)
    cv2.putText(frame, text, (x, y), font, scale, yellow, thick)

    if source == 'video':
        timestamp = datetime.min + timedelta(seconds=(frameNum / fps))
        elapsedTime = datetime.strftime(timestamp, '%H:%M:%S')
        timestamp = datetime.min + timedelta(seconds=(frameCount / fps))
        totalTime = datetime.strftime(timestamp, '%H:%M:%S')

        text = 'Time: {}/{}'.format(elapsedTime, totalTime)
        size, _ = cv2.getTextSize(text, font, scale, thick)
        y = frame.shape[0] - 5
        cv2.putText(frame, text, (x, y), font, scale, black, glow)
        cv2.putText(frame, text, (x, y), font, scale, yellow, thick)

    # Print the help message
    text = 'Press F1 for help'
    size, _ = cv2.getTextSize(text, font, scale, thick)
    x = frame.shape[1] - size[0] - 5
    y = frame.shape[0] - size[1] + 5
    cv2.putText(frame, text, (x, y), font, scale, black, glow)
    cv2.putText(frame, text, (x, y), font, scale, yellow, thick)

#---------------------------------------------
def showHelp(windowTitle, shape):
    """
    Displays an image with helping text.

    Parameters
    ----------
    windowTitle: str
        Title of the window where to display the help
    shape: tuple
        Height and width of the window to create the help image.
    """

    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.0
    thick = 1

    # Color settings
    black = (0, 0, 0)
    red = (0, 0, 255)

    # Create the background image
    image = np.ones((shape[0], shape[1], 3)) * 255

    # The help text is printed in one line per item in this list
    helpText = [
    'Controls:',
    '-----------------------------------------------',
    '[q] or [ESC]: quits from the application.',
    '[p]: toggles paused/playing the video/webcam input.',
    '[r]: restarts the video playback (video input only).',
    '[left/right arrow]: displays the previous/next frame (video input only).',
    '[page-up/down]: rewinds/fast forwards by 10 seconds (video input only).',
    ' ',
    ' ',
    'Press any key to close this window...'
    ]

    # Print the controls help text
    xCenter = image.shape[1] // 2
    yCenter = image.shape[0] // 2

    margin = 20 # between-lines margin in pixels
    textWidth = 0
    textHeight = margin * (len(helpText) - 1)
    lineHeight = 0
    for line in helpText:
        size, _ = cv2.getTextSize(line, font, scale, thick)
        textHeight += size[1]
        textWidth = size[0] if size[0] > textWidth else textWidth
        lineHeight = size[1] if size[1] > lineHeight else lineHeight

    x = xCenter - textWidth // 2
    y = yCenter - textHeight // 2

    for line in helpText:
        cv2.putText(image, line, (x, y), font, scale, black, thick * 3)
        cv2.putText(image, line, (x, y), font, scale, red, thick)
        y += margin + lineHeight

    # Show the image and wait for a key press
    cv2.imshow(windowTitle, image)
    cv2.waitKey(0)

#---------------------------------------------
def parseCommandLine(argv):
    """
    Parse the command line of this utility application.

    This function uses the argparse package to handle the command line
    arguments. In case of command line errors, the application will be
    automatically terminated.

    Parameters
    ------
    argv: list of str
        Arguments received from the command line.

    Returns
    ------
    object
        Object with the parsed arguments as attributes (refer to the
        documentation of the argparse package for details)

    """
    parser = argparse.ArgumentParser(description='Tests the face and emotion '
                                        'detector on a video file input.')

    parser.add_argument('source', nargs='?', const='Yes',
                        choices=['video', 'cam'], default='cam',
                        help='Indicate the source of the input images for '
                        'the detectors: "video" for a video file or '
                        '"cam" for a webcam. The default is "cam".')

    parser.add_argument('-f', '--file', metavar='<name>',
                        help='Name of the video file to use, if the source is '
                        '"video". The supported formats depend on the codecs '
                        'installed in the operating system.')

    parser.add_argument('-i', '--id', metavar='<number>', default=0, type=int,
                        help='Numerical id of the webcam to use, if the source '
                        'is "cam". The default is 0.')


    args = parser.parse_args()

    if args.source == 'video' and args.file is None:
        parser.error('-f is required when source is "video"')

    return args

#---------------------------------------------
# namespace verification for invoking main
#---------------------------------------------
if __name__ == '__main__':
    main(sys.argv[1:])