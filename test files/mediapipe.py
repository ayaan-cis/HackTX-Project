import cv2
import mediapipe as mp
import numpy as np
import time
# from playsound import playsound
import os


# Function to calculate the angle between three points
def calculate_angle(point1, point2, point3):
    """
    Calculate the angle between three points (e.g., joint coordinates).
    The angle is measured at point2, which is the middle point.

    Args:
    - point1: tuple (x, y) - The coordinates of the first point
    - point2: tuple (x, y) - The coordinates of the middle point
    - point3: tuple (x, y) - The coordinates of the third point

    Returns:
    - angle: float - The angle in degrees
    """
    # Convert points to vectors
    vector1 = np.array([point1[0] - point2[0], point1[1] - point2[1]])
    vector2 = np.array([point3[0] - point2[0], point3[1] - point2[1]])

    # Calculate the dot product and the magnitude of vectors
    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    # Safeguard against zero division
    if magnitude1 == 0 or magnitude2 == 0:
        print("Warning: One of the vectors has zero magnitude.")
        return 0.0

    # Calculate the angle in radians and then convert to degrees
    angle_rad = np.arccos(dot_product / (magnitude1 * magnitude2))
    angle_deg = np.degrees(angle_rad)

    # Handle cases where numerical instability might give NaN
    if np.isnan(angle_deg):
        return 0.0

    return angle_deg


# Initialize MediaPipe Pose and webcam
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Convert the frame color from BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    # Draw the pose landmarks on the frame
    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                              landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    # Check if landmarks are detected
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Convert landmark positions from normalized to pixel coordinates
        hip = (int(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * frame.shape[1]),
               int(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * frame.shape[0]))
        knee = (int(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * frame.shape[1]),
                int(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * frame.shape[0]))
        ankle = (int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * frame.shape[1]),
                 int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * frame.shape[0]))
        foot = (int(landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x * frame.shape[1]),
                int(landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y * frame.shape[0]))

        # Calculate the angles at the knee, hip, and ankle
        knee_angle = calculate_angle(hip, knee, ankle)
        hip_angle = calculate_angle((hip[0], hip[1] - 50), hip,
                                    knee)  # Using a vertical point above the hip for calculation
        ankle_angle = calculate_angle(knee, ankle, foot)

        # Print the angles for debugging
        print("Knee angle:", knee_angle)
        print("Hip angle:", hip_angle)
        print("Ankle angle:", ankle_angle)

        # Display the angles on the frame
        cv2.putText(frame, f'Knee: {int(knee_angle)}',
                    tuple(np.multiply(knee, [frame.shape[1], frame.shape[0]]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(frame, f'Hip: {int(hip_angle)}',
                    tuple(np.multiply(hip, [frame.shape[1], frame.shape[0]]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(frame, f'Ankle: {int(ankle_angle)}',
                    tuple(np.multiply(ankle, [frame.shape[1], frame.shape[0]]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Posture Corrector', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
