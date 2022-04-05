from flask import Flask, Response,render_template
from camera import VideoCamera
import cv2

app = Flask(__name__)

cap = cv2.VideoCapture(0)

@app.route("/")
def index():
    return render_template('index.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame != b'0':
            yield(b'--frame\r\n'
                b'Content-type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else:
            yield(b'--frame\r\n'
                b'Content-type: image/jpeg\r\n\r\n' b'\r\n\r\n')

@app.route("/video_feed")
def video_feed():
    return Response(gen(VideoCamera(cap)), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=2204, debug = False)