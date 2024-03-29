# LPD-for-RPi

This is a Project to perform License Plate Detection with inference being performed on a Raspberry Pi 3B.
The RPi detects License Plates using ML and sends inferred frames to a backend server for analysis.

The RPi is connected to an Arduino which acts as steering for the motor, based on inferred data.

Three methods of inference on a RPi have been posted
All three folders contain the testing code for each specific model

The final chosen inference method is ONNX
The script is located in the ONNX folder

The final report submission has also been provided.

# Inference Backend
Backend ML Server hosted in HuggingFace Spaces
<br>
https://huggingface.co/spaces/ll753/FlaskMLBackendLPR

# View Captured Information
Captured frame can be viewed on a hosted website
https://nextjs-lpr.vercel.app/
<br>
Code for Website:
https://github.com/lorocks/NextJS-LPR



# To use Inference in RPi, ML Activation

## Inference with Arduino
The 'serial_object_detection.py' script is to be used with an Arduino connected to the RPi

## Inference without Arduino
The 'final_object_detection.py' script is the ML Activation method to be used without the Arduino

## Installing RPi Dependencies
To install dependencies on the RPi use
<br>
https://github.com/lorocks/ML-Installation-for-RPi-64bit-3.9-Python


# Drive link for ONNX 
https://drive.google.com/file/d/1BWH24eYsWIBKp-Q47n6olxqjvERsa8CF/view?usp=sharing

# For Motor Working
Install PySerial
````
pip3 install pyserial
````

Then embed motorLPD.py to the detect
