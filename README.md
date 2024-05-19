# Face-Recognition-wLive

## Face Recognition with OpenCV and Face_Recognition

This repository contains a Python script that leverages OpenCV and the face_recognition library to identify faces in an image. The script performs face detection and recognition, highlighting identified faces and displaying their names and confidence scores.

### Key Features

* Face Detection: Identifies faces in an image using a pre-trained model.
* Face Recognition: Compares detected faces with known face encodings to recognize individuals.
* Visual Output: Displays the results with bounding boxes around faces and labels indicating the recognized name and confidence score.

### Requirements

⚹ face_recognition
⚹ opencv-python
⚹ numpy
⚹ matplotlib


You can install these dependencies using pip:

```
pip install face_recognition opencv-python numpy matplotlib

```

### Usage

Prepare Your Images:

Place the image of the person to be recognized in the Images/actualPhoto directory.
Place the target image (where faces will be detected and recognized) in the Images/target/Photo directory.

```
data = {
    image = face_recognition.load_image_file("Images/actualPhoto")

    frame = cv2.imread("Images/target/Photo")
}

```

Run the Script: Execute the script to start the video capture and face recognition process.

```
 python face_recognition.py
 
```