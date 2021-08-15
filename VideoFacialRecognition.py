import cv2
from random import randrange

def detectAndDisplay(frame):
    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.equalizeHist(gray_frame)

    # Detect trained faces no matter the scale
    face_location = trained_facial_data.detectMultiScale(gray_frame)

    for (x, y, w, h) in face_location:
        #cv2.rectangle(frame, (x, y), (x + w, y + h), (256, 256, 256), 5)
        # Draw circles around all detected faces
        face_center = (x + w // 2, y + h // 2)
        face_radius = int(round((w + h) * 0.3))
        cv2.circle(frame, face_center, face_radius , (256, 256, 256), 4)

        # In each face, detect eyes
        targetArea = gray_frame[y:y + h, x:x + w]
        eyes = trained_eye_data.detectMultiScale(targetArea)

        for (x2, y2, w2, h2) in eyes:
            # Draw triangles around all detected eyes
            eye_center = (x + x2 + w2 // 2, y + y2 + h2 // 2)
            radius = int(round((w2 + h2) * 0.3))
            frame = cv2.circle(frame, eye_center, radius, (255, 0, 0), 4)
            # cv2.rectangle(frame, (x2, y2), (x2 + w2, y2 + h2), (256, 256, 256), 5)

    # Display video
    cv2.imshow('Face Display', frame)
    cv2.waitKey(1)

# Load pre-trained data on face frontals from opencv (haar cascade algorithm)
facePath = r'C:\Users\avery\Documents\Stony Brook\2020-2021\Summer 2021\AI\haarcascade_frontalface_default.xml'
trained_facial_data = cv2.CascadeClassifier(facePath)

# Load pre-trained data on frontal eyes
eyesPath = r'C:\Users\avery\Documents\Stony Brook\2020-2021\Summer 2021\AI\haarcascade_eye.xml'
trained_eye_data = cv2.CascadeClassifier(eyesPath)

# Import video from webcam
#vidPATH = r'C:\Users\avery\Videos\LIRR.MOV'
# Pass 0 into VideoCapture to capture live webcam
video = cv2.VideoCapture(0)
if not video.isOpened:
    print('Error opening video capture')
    exit(0)

# Loop over frames of video
while True:
    # Read the current frame
    successful_frame_read, frame = video.read()

    if frame is None:
        print('No captured frame')
        break

    # Call detection function
    detectAndDisplay(frame)



