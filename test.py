import Flask
import cv2
import mediapipe as mp
import numpy as np
from playsound import playsound
import threading

app = Flask(__name__)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Camera capture
camera = cv2.VideoCapture(0)


# Function to calculate angles between three points
def calculate_angle(point1, point2, point3):
    vector1 = np.array([point1[0] - point2[0], point1[1] - point2[1]])
    vector2 = np.array([point3[0] - point2[0], point3[1] - point2[1]])

    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    angle_rad = np.arccos(dot_product / (magnitude1 * magnitude2))
    angle_deg = np.degrees(angle_rad)

    return angle_deg if not np.isnan(angle_deg) else 0.0


# Function to play sound in a separate thread
def play_sound_async(sound_path):
    threading.Thread(target=playsound, args=(sound_path,), daemon=True).start()


# Function to generate video frames
def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

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

                # Calculate angles
                knee_angle = calculate_angle(hip, knee, ankle)
                hip_angle = calculate_angle((hip[0], hip[1] - 50), hip, knee)
                ankle_angle = calculate_angle(knee, ankle, foot)

                # Play sound if an angle exceeds a specific threshold
                if knee_angle > 150:  # Example condition
                    play_sound_async('alert.mp3')  # Replace 'alert.mp3' with the path to your sound file

                # Draw angles on the frame
                cv2.putText(frame, f'Knee: {int(knee_angle)}', knee, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                            cv2.LINE_AA)
                cv2.putText(frame, f'Hip: {int(hip_angle)}', hip, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                            cv2.LINE_AA)

