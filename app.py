from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np

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

                # Convert landmark positions to pixel coordinates
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

                # Display the angles on the frame
                cv2.putText(frame, f'Knee: {int(knee_angle)}', knee, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                            cv2.LINE_AA)
                cv2.putText(frame, f'Hip: {int(hip_angle)}', hip, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                            cv2.LINE_AA)
                cv2.putText(frame, f'Ankle: {int(ankle_angle)}', ankle, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                            cv2.LINE_AA)

            # Encode the frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
