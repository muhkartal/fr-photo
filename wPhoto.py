import face_recognition
import cv2
import numpy as np
from matplotlib import pyplot as plt

def imshow(title="Image", image=None, size=10):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w / h
    plt.figure(figsize=(size * aspect_ratio, size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()

image = face_recognition.load_image_file("Images/actualPhoto")
muhammed_face_encoding = face_recognition.face_encodings(image)[0]

known_face_encodings = [
    muhammed_face_encoding
]
known_face_names = [
    "muhammed",
]

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

frame = cv2.imread("Images/target/Photo")

rgb_frame = frame[:, :, ::-1]

if process_this_frame:
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []
    face_distances = face_recognition.face_distance(np.array(known_face_encodings), np.array(face_encodings))
    best_match_index = np.argmin(face_distances)
    for face_encoding, face_distance in zip(face_encodings, face_distances):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.9)
        name = "Unknown"
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)
        confidence = 1 - face_distance
        confidence_text = f"Netlik: {confidence:.2f}"
        print(f"{name} - {confidence_text}")

for (top, right, bottom, left), name in zip(face_locations, face_names):
    t = 8
    l = 50
    rt = 3
        
    top1 = top + bottom
    bottom1 = left + right


    cv2.rectangle(frame, (left, top), (right, bottom), (0,  0, 255), rt)

    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), rt)

    cv2.line(frame, (left, top), (left + l, top), (0, 0, 255), t) 
    cv2.line(frame, (left, top), (left, top + l), (0, 0, 255), t)  

    cv2.line(frame, (right, top), (right - l, top), (0, 0, 255), t)  
    cv2.line(frame, (right, top), (right, top + l), (0, 0, 255), t) 

    cv2.line(frame, (left, bottom), (left + l, bottom), (0, 0, 255), t) 
    cv2.line(frame, (left, bottom), (left, bottom - l), (0, 0, 255), t) 

    cv2.line(frame, (right, bottom), (right - l, bottom), (0, 0, 255), t)
    cv2.line(frame, (right, bottom), (right, bottom - l), (0, 0, 255), t)  
                
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame, name, (left + 6, bottom - 40), font, 0.8, (255, 255, 255), 1)
    confidence = 1 - face_distances[best_match_index]
    confidence_text = f"Netlik: {confidence:.2f}"
    cv2.putText(frame, confidence_text, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)

imshow('Face Recognition', frame)
