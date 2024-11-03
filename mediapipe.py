import cv2
import mediapipe as mp
import numpy as np
import time
# from playsound import playsound
import os

def calculate_angle(point1, point2, point3):
    # Convert points to vectors
    vector1 = np.array([point1[0] - point2[0], point1[1] - point2[1]])
    vector2 = np.array([point3[0] - point2[0], point3[1] - point2[1]])

    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    if magnitude1 == 0 or magnitude2 == 0:
        print("Warning: One of the vectors has zero magnitude.")
        return 0.0

    angle_rad = np.arccos(dot_product / (magnitude1 * magnitude2))
    angle_deg = np.degrees(angle_rad)

    if np.isnan(angle_deg):
        return 0.0

    return angle_deg


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                              landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Convert landmark positions from normalized to pixel coordinates
        hip = (int(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * frame.shape[1]),
               int(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * frame.shape[0]))
        knee = (int(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * frame.shape[1]),
                int(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * frame.shape[0]))
        ankle = (int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * frame.shape[1]),
                 int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * frame.shape[0]))

        print("Hip coordinates:", hip)
        print("Knee coordinates:", knee)
        print("Ankle coordinates:", ankle)

        angle = calculate_angle(hip, knee, ankle)
        print("Calculated angle:", angle)

        cv2.putText(frame, str(int(angle)),
                    tuple(np.multiply(knee, [frame.shape[1], frame.shape[0]]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Posture Corrector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


