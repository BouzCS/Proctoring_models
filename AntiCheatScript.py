from proctoring import get_analysis, yolov3_model_v3_path
import cv2

import base64

def model(img_64_encode):
    # insert the path of yolov3 model [mandatory]
    yolov3_model_v3_path("./models/yolov3.weights")


    # Encode Image to base64 :
    proctorData = get_analysis(img_64_encode, "./models/shape_predictor_68_face_landmarks.dat")
    return proctorData


mobile_c=per_c=mov_up_do_c=mov_r_l_c=e_m_c=0

def counter(dict):
    global per_c,e_m_c,mobile_c,mov_up_do_c,mov_r_l_c
    if not dict["mob_status"]=="Not Mobile Phone detected":
        mobile_c+=1
        
    if not (dict["user_move2"]=='Looking center'):
        mov_r_l_c+=1
        
    
    return {'mob_status_counter': mobile_c,'user_move2_counter': mov_r_l_c}
    


cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX 
while(True):
    ret, img = cap.read()
    if ret == True:
        img = cv2.resize(img, None, fx=0.5, fy=0.5)
                
        # Convert captured image to JPG
        retval, buffer = cv2.imencode('.jpg', img)

        # Convert to base64 encoding and show start of data
        img_64_encode = base64.b64encode(buffer)
        
        
                
        status = model(img_64_encode)
        print(status)
        print("#"*50)
        print(counter(status))
        
        cv2.imshow("Main", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
        
cap.release()
cv2.destroyAllWindows()