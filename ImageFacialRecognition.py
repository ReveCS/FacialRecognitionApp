import cv2
from random import randrange

# Load pre-trained data on face frontals from opencv (haar cascade algorithm)
cscdPath = r'C:\Users\avery\Documents\Stony Brook\2020-2021\Summer 2021\AI\haarcascade_frontalface_default.xml'
trained_facial_data = cv2.CascadeClassifier(cscdPath)

# Choose an image to detect faces in
imgPath = r'C:\Users\avery\Pictures\Parasite.jpg'
img = cv2.imread(imgPath)

# Convert image to grayscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect trained faces no matter the scale
face_location = trained_facial_data.detectMultiScale(grayscaled_img)
print(face_location)

# Draw squares around all detected faces
for (x, y, w, h) in face_location:
    cv2.rectangle(img, (x, y), (x + w, y + h), (randrange(256), randrange(256), randrange(256)), 5)

# Show chosen image
cv2.imshow('Face Display', img)
cv2.waitKey()