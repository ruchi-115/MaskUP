import cv2
import numpy as np

trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
mouth_cascade = cv2.CascadeClassifier('mouth.xml')
eyes_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Choose an image to detect faces in
#img = cv2.imread('RDJ.jpg')

#to capture video from webcam.
webcam = cv2.VideoCapture(0)
key = cv2.waitKey(1)
weared_mask_font_color = (0, 255, 0)
not_weared_mask_font_color = (0, 0, 255)
weared_mask = "Thank You for wearing MASK"
not_weared_mask = "Please wear a MASK "
font = cv2.FONT_HERSHEY_DUPLEX
#iterate forever over frames
while True:
    successful_frame_read, frame = webcam.read()
    frame = cv2.flip(frame,1)
    #Must convert to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
    mouth_rects = mouth_cascade.detectMultiScale(grayscaled_img, 1.5, 5)
    eyes_rects = eyes_cascade.detectMultiScale(grayscaled_img,1.5,5)
        
    # Draw rectangles around the faces
    if(len(face_coordinates) == 0):
        cv2.putText(frame, "No face found...", (30,30), font, 1,(0,0,0), 2, cv2.LINE_AA)
    else:
        for (x, y, h, w) in  face_coordinates:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
            
            #Check for mask
            if(len(mouth_rects) == 0 and len(eyes_rects) != 0):
                    cv2.rectangle(frame, (x, y), (x+w ,y+h),(weared_mask_font_color), 5)
                    cv2.putText(frame, weared_mask, (30,30), font,1, weared_mask_font_color, 2, cv2.LINE_AA)
            else:
                for (mx, my, mw, mh) in mouth_rects:
                    if(y < my < y + h):
                        cv2.rectangle(frame, (x, y), (x+w ,y+h),(not_weared_mask_font_color), 5)
                        cv2.putText(frame, not_weared_mask, (30,30), font, 1, not_weared_mask_font_color, 2, cv2.LINE_AA)
                        break
    cv2.imshow('Face detector', frame)
    cv2.waitKey(1)

    ### Stop if Q key is pressed
    if key== 81 or key==113:
        webcam.release()
        break
### Release the VideoCapture object

