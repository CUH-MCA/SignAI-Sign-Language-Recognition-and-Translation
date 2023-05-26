from flask import Flask, render_template, Response
import cv2
import mediapipe as mp

app = Flask(__name__)

def generate_frames():
    mp_hand = mp.solutions.hands
    hands = mp_hand.Hands()
    mp_drawing_utils = mp.solutions.drawing_utils
    cap=cv2.VideoCapture(0)
    while True:
            
        ## read the camera frame
        success, frame = cap.read()
        if not success:
            break
        else:
            result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if result.multi_hand_landmarks:
                for hand_landmark in result.multi_hand_landmarks:
                    mp_drawing_utils.draw_landmarks(frame, hand_landmark, mp_hand.HAND_CONNECTIONS)

            ret,buffer=cv2.imencode('.jpg',frame)
            
            frame=buffer.tobytes()

        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)