import cv2
from random import randrange

# Load pre-trained data on face frontals from opencv (haar cascade algorithm)
cscdPath = r'C:\Users\avery\Documents\Stony Brook\2020-2021\Summer 2021\AI\haarcascade_frontalface_default.xml'
trained_facial_data = cv2.CascadeClassifier(cscdPath)

# Import video from webcam
vidPATH = r'C:\Users\avery\Videos\LIRR.MOV'
# Pass 0 into VideoCapture to capture live webcam
video = cv2.VideoCapture(vidPATH)

# Loop over frames of video
while True:
    # Read the current frame
    successful_frame_read, frame = video.read()

    # Convert frame to grayscale
    grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect trained faces no matter the scale
    face_location = trained_facial_data.detectMultiScale(grayscaled_frame)

    # Draw squares around all detected faces
    for (x, y, w, h) in face_location:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (256, 256, 256), 5)

    # Display video
    cv2.imshow('Face Display', frame)
    cv2.waitKey(1)

