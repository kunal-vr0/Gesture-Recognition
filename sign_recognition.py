import cv2
import mediapipe as mp
import numpy as np
from math import acos
import pickle
width = 1080
height = 720
#================================================================================================================================================================================
class HandDet():
    def __init__(self, max_hands = 2, still = False, modelComplexity = 1, tol1 = 0.5, tol2 = 0.5):
        self.max_hands = max_hands
        self.still = still
        self.modelComplexity = modelComplexity
        self.tol1 = tol1
        self.tol2 = tol2 
        self.mpHands = mp.solutions.hands
        self.mpDraw = mp.solutions.drawing_utils
        self.mpStyle = mp.solutions.drawing_styles
        self.myHands = self.mpHands.Hands(self.still, self.max_hands, self.modelComplexity, self.tol1, self.tol2)

    def findHands(self, frame, draw = True):
        RGBframe = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        results = self.myHands.process(RGBframe)
        bothHands = []
        if results.multi_hand_landmarks:
            for HandLMs in results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, HandLMs, self.mpHands.HAND_CONNECTIONS, self.mpStyle.get_default_hand_landmarks_style(), self.mpStyle.get_default_hand_connections_style()) 
                oneHand = []
                for LM in HandLMs.landmark:
                    x = int(width*LM.x)
                    y = int(height*LM.y)
                    oneHand.append((x,y))
                bothHands.append(oneHand)
        return bothHands
#=================================================================================================================================================================================
#distance
def distance(pt1, pt2):
    x = pt1[0] - pt2[0]
    y = pt1[1] - pt2[1]
    r = x**2 + y**2
    r = r**0.5
    return r
#====================================================================================================================================================================================
#angle
def angle(x,y,z):
    if x!=0 and y!=0 and z!=0:
        cos = (x**2 + y**2 - z**2)/(2*x*y)
        if cos > 1:
            cos = 1
        if cos < -1:
            cos = -1
    else:
        return 0 
    #sin = (1-cos**2)**0.5
    return acos(cos)
#==================================================================================================================================================================================
#reading gesture
def readGesture(allpts):
    pts = [0,1,2,4,5,8,9,12,13,16,17,20]
    gestureData = np.zeros([12,12], dtype= np.float32)
    for i in range(12):
        for j in range(12):
            z = distance(allpts[pts[i]], allpts[pts[j]])
            x = distance(allpts[pts[i]], allpts[0])
            y = distance(allpts[pts[j]], allpts[0])
            gestureData[i][j] = angle(x,y,z)
    
    return gestureData
#=====================================================================================================================================================================================

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cam.set(cv2.CAP_PROP_FPS, 30)
cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
HandData = HandDet(1)
#=====================
tolerance = 3
#=====================
train = False
record_prompt = False
name = True
name_prompt = True
#=====================
gesture_name = ''
all_names = []
known_gestures = []
known_gest_sums = []  #==========
print("Press n To Add New Gesture")
while True:
    ig, frame = cam.read()
    frame = cv2.resize(frame, (width, height))
    frame = cv2.flip(frame, 1)
    cur_gesture = np.zeros([12,12], dtype=np.float32)
    handLM = HandData.findHands(frame)
    for LM in handLM:
        cur_gesture = readGesture(LM)
#================================================= Training the software
    if train:
        #=========================================== entering the name
        if name:
            if name_prompt:
                print("Please Do Not Exit While Recording Gesture ^^")
                print("Enter Gesture Name: ")
                name_prompt = False
            letter = cv2.waitKey(1)
            if letter == -1:
                continue
            elif letter == 13:
                all_names.append(gesture_name)
                gesture_name = ''
                name = False
            else:
                gesture_name += chr(letter)
        #============================================ name recorded
        #============================================ recording the gesture
        else:
            if record_prompt:
                print("Press t When Ready...")
                record_prompt = False
            if cv2.waitKey(1) & 0xff == ord('t'):
                known_gestures.append(cur_gesture)
                known_gest_sums.append(cur_gesture.sum())
                record_prompt = True
                name_prompt = True
                name = True
                train = False
        #============================================ recorded
#========================================================= gesture learned
#========================================================= gesture recognition
    else:
        errors = []
        for known_gesture in known_gest_sums:
            b = cur_gesture.sum()
            error = (known_gesture-b)
            errors.append(abs(error))
        index = None
        if errors:
            best_match = min(errors)
            if best_match < tolerance:
                index = errors.index(best_match)
                #print(best_match, 'Index: ',index)

        if index != None:
            cv2.putText(frame, all_names[index], (0, 80), cv2.FONT_HERSHEY_COMPLEX, 3, (0,0,255), 3)
        else :
            cv2.putText(frame, 'Unknown', (0, 40), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,255), 3)

    cv2.imshow('Frame', frame)
    cv2.moveWindow('Frame', 0,0)

    if not train:
        k = cv2.waitKey(1)
        if k == 110:   #n
            train = True
        if k == 112:   #p
            print(all_names)
            print(len(known_gest_sums))
            print(known_gest_sums)
            print(errors)
        if k == 113:   #q
            break

cam.release()
