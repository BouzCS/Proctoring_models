import face_recognition
import cv2
import numpy as np

#path : originale picture path
#new_image_path : new image to be verified path

def load_and_encode_img(path):
    
    # Load a second sample picture and learn how to recognize it.
    user_image = face_recognition.load_image_file(path)

    user_face_encoding = face_recognition.face_encodings(user_image)[0]
    return user_face_encoding

def faceverification(users_face_encoding,face_users_names,new_image_path):
# Get a reference to webcam #0 (the default one)
    image = cv2.imread(new_image_path)

    # Create arrays of known face encodings and their names
    known_face_encodings = users_face_encoding
    known_face_names = face_users_names

    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True


    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    
    return(face_names[0])
