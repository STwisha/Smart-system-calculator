# Smart-system-calculator
A Deep Learning–Powered Virtual Calculator using Webcam + Custom Hand Gesture Dataset

📌 Project Overview

This project implements a virtual calculator controlled entirely through hand gestures captured via a webcam.
Using a custom dataset of hand signs representing:
	•	Digits: 0–9
	•	Operators: +, -, *, /, ^

A Convolutional Neural Network (CNN) model (MobileNetV2) is trained to classify these gestures in real time.
The system reads the gesture sequence, forms mathematical expressions, evaluates them, and displays the result.

🧠 Key Features

✔ Gesture detection using OpenCV
✔ Real-time prediction using PyTorch
✔ Lightweight model using MobileNetV2
✔ 15-class custom dataset (0–9 + 5 operators)
✔ Webcam-based input
✔ Fully functional calculator logic
✔ Modular file structure
