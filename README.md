# Detection of Emotions
A simple detector of prototypic emotions from facial images captured in video, written in Python. Copyright (C) 2016-2017 Luiz Carlos Vieira.

This project works by first detecting a face in each frame of a video. The face region and the landmark coordinates found and then used to detect the probabilities of each prototypic emotion (happiness, sadness, anger, fear, surprise and disgust, plus the neutral expression).

[![Video with a test of this project](https://img.youtube.com/vi/uDRYr0CWSwI/0.jpg)](https://youtu.be/uDRYr0CWSwI)

(Click on this image to see a video with a test of this project - the original video is called "Human Emotions" and it was used with the kind authorization from the folks of the Imagine Pictures video production company (thank you guys!). Original URL of their video: [https://www.youtube.com/watch?v=UjZzdo2-LKE](https://www.youtube.com/watch?v=UjZzdo2-LKE))

## Details on the Implementation

The face detection is based on a Cascade Detector (Viola-Jones algorithm). The facial landmarks are located by using a deformable model adjusted to the face region previously found with the Cascade. The code uses the implementation available from the dlib library. Its trained detector model is loaded from the folder `models`.

The emotion detection is based on a Support Vector Machine (SVM) using the RBF kernel and trained from labelled images obtained from two openly available (for non commercial use) datasets:

- [The Cohn-Kanade AU-Coded Expression Database (CK+)](http://www.pitt.edu/~emotion/ck-spread.htm)
- [The 10k US Adult Faces Database](http://wilmabainbridge.com/facememorability2.html)

The code uses the SVM implementation from the Scikit-Learn package. The training and the detection is based on the responses of a bank of [Gabor filters](https://en.wikipedia.org/wiki/Gabor_filter) applied to the face region on the frame image. Only the responses under the coordinates of the facial landmarks are used. The bank of Gabor filters is implemented in file `gabor.py`. The trained detector model is also loaded from the folder `models`.

## Project Dependencies

This project depends on [Python](https://www.python.org/) (>= 3.5) and the following packages:

- [OpenCV](http://opencv.org/) (>= 3.1.0)
- [dlib](http://dlib.net/) (>= 19.1.0)
- [scikit-learn](http://scikit-learn.org/) (>= 0.18.1)
- [scikit-image](http://scikit-image.org/) (>= 0.12.0)

## Usage

The script `test.py` is used to test the detectors. Its usage is the following (which is shown by passing the argument `-h` to the command line):

>> usage: test.py [-h] [-f <name>] [-i <number>] [{video,cam}]
>> 
>> Tests the face and emotion detector on a video file input.
>> 
>> positional arguments:
>>   {video,cam}           Indicate the source of the input images for the
>>                         detectors: "video" for a video file or "cam" for a
>>                         webcam. The default is "cam".
>> 
>> optional arguments:
>>   -h, --help            show this help message and exit
>>   -f <name>, --file <name>
>>                         Name of the video file to use, if the source is
>>                         "video". The supported formats depend on the codecs
>>                         installed in the operating system.
>>   -i <number>, --id <number>
>>                         Numerical id of the webcam to use, if the source is
>>                         "cam". The default is 0.

## License

The code is licensed under MIT license and can be used as desired. Only the trained models (the files in `model` folder) can not be used for commercial purposes due to the limitations on the licenses of the datasets used to train them.

If you want to use the code for commercial purposes, you must train your own model from your own image datasets labelled with the prototypic emotions.

## Known Issues

The datasets used for training the emotion detector had many more samples of Neutral and Happiness expressions than the other expressions, causing the SVM classifier produced to be unbalanced (i.e. to prioritize the detection of these two expressions). A procedure was applied to try to minimize that effect (the parameter `C` of the trained SVM was weighted by the number of samples of each class, [as described here](http://scikit-learn.org/stable/modules/svm.html#unbalanced-problems)), but still the results were not as good as they could have been if a balanced database of labelled images were used.

The emotion detection is based only on individual frames. That is, the classifier does not use any temporal information, like the estimated probabilities of emotions from previous frames. If a [structured prediction](https://en.wikipedia.org/wiki/Structured_prediction) approach was used, the results would have been better because the expression of emotions have an intrinsic structure: they progress from onset, to apex and then to offset. The structured approach was not used in this version because it is difficult to find datasets with labelled sequences of emotions captured in nature (CK+ have sequences, but in a very small number of samples).

The bank of Gabor filters is constructed using the functions from the scikit-image package instead of the OpenCV library. The reason is because OpenCV's implementations only produces the real part of the filter response, and the solution used in this project requires both the real and imaginary parts of the Gabor responses. Nonetheless, the application of the filter uses convolution, which is performed using OpenCV. This application is the most computational intensive task (convolution of 32 filters for each frame of a video), which causes the processing to run with a very low frame rate. The OpenCV library has GPU specific functions that are intended to parallelize some operations, but [as far as I known they can only be used in C++ (and not in the binding for Python)](http://answers.opencv.org/question/35749/using-opencv-gpu-functions-in-python/).