from func import *

from retinaface import RetinaFace
from deepface import DeepFace

import cv2
import json

image_dir = "3.jpg"

pics1 = "1.jpg"
pics2 = "2.jpg"

def information(pics):
    # get information (score, coordinates face, landmark)
    resp = RetinaFace.detect_faces(pics)
    print(resp)

    # # Convert NumPy types to Python native types
    # resp_converted = convert_np_types(resp)
    #
    # # Print the response as a nicely formatted JSON string
    # print(json.dumps(resp_converted, indent=4))


# show image with face detection

def face_detection(pics):
    image_original = cv2.imread(pics)
    height, width = image_original.shape[:2]

    image = cv2.resize(image_original, (int(width * 0.5), int(height * 0.5)))

    resp = RetinaFace.detect_faces(image)

    for face_key in resp:
        # Get the facial area (bounding box coordinates) of each detected face
        facial_area = resp[face_key]['facial_area']
        x1, y1, x2, y2 = facial_area  # Extract the coordinates

        # Draw a rectangle (bounding box) around the face
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green color with thickness 2

    # Step 5: Show the image with bounding boxes
    cv2.imshow("Face Detection", image)
    # Wait for the 'q' key to be pressed to close the window
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Check if 'q' is pressed
            break

    # Close all OpenCV windows
    cv2.destroyAllWindows()

def get_landmark(pics):
    image_original = cv2.imread(pics)
    height, width = image_original.shape[:2]

    image = cv2.resize(image_original, (int(width * 0.5), int(height * 0.5)))

    resp = RetinaFace.detect_faces(image)

    for face_key in resp:
        landmarks = resp[face_key]['landmarks']
        for landmark in landmarks.values():
            # Draw a small circle for each landmark
            cv2.circle(image, (int(landmark[0]), int(landmark[1])), 2, (0, 0, 255), -1)  # Red color

    # Step 5: Show the image with bounding boxes
    cv2.imshow("Face Detection", image)
    # Wait for the 'q' key to be pressed to close the window
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Check if 'q' is pressed
            break

    # Close all OpenCV windows
    cv2.destroyAllWindows()

def detect_realtime():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)  # 0 for default webcam, or use other index if needed

    while True:
        # Capture frame-by-frame from the webcam
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame")
            break

        # Use RetinaFace to detect faces in the frame
        resp = RetinaFace.detect_faces(frame)
        #
        # # If faces are detected, draw bounding boxes around them
        if resp:
            for face_key in resp:
                # Get the facial area (bounding box coordinates) of each detected face
                facial_area = resp[face_key]['facial_area']
                x1, y1, x2, y2 = facial_area  # Extract the coordinates

                # Draw a rectangle (bounding box) around the face
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green color with thickness 2

        # Display the frame with bounding boxes in a window
        cv2.imshow("Real-time Face Detection", frame)

        # Press 'q' to exit the webcam feed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

def alignment(pics):
    # Extract aligned faces using RetinaFace
    faces = RetinaFace.extract_faces(img_path=pics, align=True)

    # Read the original image
    pic = cv2.imread(pics)

    # Loop through all extracted faces
    for face in faces:
        # Display the original image
        cv2.imshow("Face Original", pic)

        # Display the aligned face
        face_bgr = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
        cv2.imshow("Face Alignment", face_bgr)

        # Wait for the 'q' key to be pressed to close the window
        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Check if 'q' is pressed
                break

    # Close all OpenCV windows
    cv2.destroyAllWindows()

def recognition(pic1, pic2):
    # Perform face recognition using DeepFace
    obj = DeepFace.verify(pic1, pic2, model_name='ArcFace', detector_backend='retinaface')

    # Convert NumPy types to Python native types
    obj_converted = convert_np_types(obj)

    # Print the response as a nicely formatted JSON string
    print(json.dumps(obj_converted, indent=4))
    print(obj)

    # Load the images using OpenCV
    img1 = cv2.imread(pic1)
    img2 = cv2.imread(pic2)

    # Extract the facial areas from the result
    facial_area1 = obj['facial_areas']['img1']
    facial_area2 = obj['facial_areas']['img2']

    # Get verification result and similarity score
    verified = obj['verified']
    distance = obj['distance']

    # Draw rectangles around faces if the images are verified as the same person
    if verified:
        verification_text = f"Verified: True (Distance: {distance:.2f})"
        color = (0, 255, 0)  # Green for verified
    else:
        verification_text = f"Verified: False (Distance: {distance:.2f})"
        color = (0, 0, 255)  # Red for non-verified

    # Draw rectangle on img1
    x1, y1, w1, h1 = facial_area1['x'], facial_area1['y'], facial_area1['w'], facial_area1['h']
    img1 = cv2.rectangle(img1, (x1, y1), (x1 + w1, y1 + h1), color, 2)

    # Draw rectangle on img2
    x2, y2, w2, h2 = facial_area2['x'], facial_area2['y'], facial_area2['w'], facial_area2['h']
    img2 = cv2.rectangle(img2, (x2, y2), (x2 + w2, y2 + h2), color, 2)

    # Put text on img1
    cv2.putText(img1, verification_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(img2, verification_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Display both images with rectangles and accuracy
    cv2.imshow("Image 1 with Face Detection and Accuracy", img1)
    cv2.imshow("Image 2 with Face Detection and Accuracy", img2)

    # Wait for key press to close the windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def menu():
    print('select options')
    print('1. information')
    print('2. face detect')
    print('3. landmark')
    print('4. detect real-time using webcam')
    print('5. alignment')
    print('6. recognition')

    option = int(input())

    if option == 1:
        information(image_dir)
    elif option == 2:
        face_detection(image_dir)
    elif option == 3:
        get_landmark(image_dir)
    elif option == 4:
        detect_realtime()
    elif option == 5:
        alignment(image_dir)
    elif option == 6:
        recognition(pics1, pics2)
    else:
        print('not support')

if __name__ == '__main__':
    menu()