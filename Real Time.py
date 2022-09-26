import cv2

cap = cv2.VideoCapture(0)
    
while cap.isOpened():
    ret, frame = cap.read()

    # Recolor Feed
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False        

    # Model Call :


    # Recolor image back to BGR for rendering
    image.flags.writeable = True   
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


    # 2. Right hand
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                mp_drawing.DrawingSpec(color=(222, 160, 87), thickness=2, circle_radius=5),
                                mp_drawing.DrawingSpec(color=(252, 255, 231), thickness=2, circle_radius=3)
                                )

    # 3. Left Hand
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                )

    # Export coordinates
    try:
        # Extract Pose landmarks
        # right = results.right_hand_landmarks.landmark
        # right_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in right]).flatten())

        # Extract Face landmarks
        # face = results.face_landmarks.landmark
        # face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())

        # landmark list
        hand_landmarks_right = results.right_hand_landmarks
        hand_landmarks_left = results.left_hand_landmarks

        # Landmark calculation
        landmark_list_right = calc_landmark_list(image, hand_landmarks_right)
        landmark_list_left = calc_landmark_list(image, hand_landmarks_left)

        # Conversion to relative coordinates / normalized coordinates
        pre_processed_landmark_list_right = pre_process_landmark(landmark_list_right)
        pre_processed_landmark_list_left = pre_process_landmark(landmark_list_left)

        # Concate rows
        # row = list(np.array(pre_processed_landmark_list).flatten()) # +face_row
        row = pre_processed_landmark_list_right+pre_processed_landmark_list_left

        


        # Concate rows
        # row = right_row # +face_row

    #             # Append class name 
    #             row.insert(0, class_name)

    #             # Export to CSV
    #             with open('coords.csv', mode='a', newline='') as f:
    #                 csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #                 csv_writer.writerow(row) 

        # Make Detections
        X = pd.DataFrame([row])
        body_language_class = model.predict(X)[0]
        body_language_prob = model.predict_proba(X)[0]
        print(body_language_class, body_language_prob)

        # Grab ear coords
        coords = tuple(np.multiply(
                        np.array(
                            (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x, 
                                results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y))
                    , [640,480]).astype(int))

        cv2.rectangle(image, 
                        (coords[0], coords[1]+5), 
                        (coords[0]+len(body_language_class)*20, coords[1]-30), 
                        (245, 117, 16), -1)
        cv2.putText(image, body_language_class, coords, 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Get status box
        cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)

        # Display Class
        cv2.putText(image, 'CLASS'
                    , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, body_language_class.split(' ')[0]
                    , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Display Probability
        cv2.putText(image, 'PROB'
                    , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)],2))
                    , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    except:
        pass
                    
    cv2.imshow('Raw Webcam Feed', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()