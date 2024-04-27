from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

from util import get_parking_spots_bboxes, empty_or_not

app = Flask(__name__)

def calc_diff(im1, im2):
    return np.abs(np.mean(im1) - np.mean(im2))

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
@app.route('/login1')
def login():
    return render_template('login1.html')
@app.route('/upload', methods=['GET', 'POST'])
def upload_files():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'image' not in request.files or 'video' not in request.files:
            return redirect(request.url)
        image_file = request.files['image']
        video_file = request.files['video']
        # if user does not select file, browser also
        # submit an empty part without filename
        if image_file.filename == '' or video_file.filename == '':
            return redirect(request.url)
        if image_file and allowed_file(image_file.filename) and video_file and allowed_file(video_file.filename):
            image_filename = secure_filename(image_file.filename)
            video_filename = secure_filename(video_file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
            image_file.save(image_path)
            video_file.save(video_path)
            mask = cv2.imread(image_path, 0)
            cap = cv2.VideoCapture(video_path)
            # Rest of your code here...
            connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
            print(connected_components)

            spots = get_parking_spots_bboxes(connected_components)

            spots_status = [None for j in spots]
            diffs = [None for j in spots]

            previous_frame = None

            frame_nmr = 0
            ret = True
            step = 30
            while ret:
                ret, frame = cap.read()

                if frame_nmr % step == 0 and previous_frame is not None:
                    for spot_indx, spot in enumerate(spots):
                        x1, y1, w, h = spot

                        spot_crop = frame[y1:y1 + h, x1:x1 + w, :]

                        diffs[spot_indx] = calc_diff(spot_crop, previous_frame[y1:y1 + h, x1:x1 + w, :])

                    print([diffs[j] for j in np.argsort(diffs)][::-1])

                if frame_nmr % step == 0:
                    if previous_frame is None:
                        arr_ = range(len(spots))
                    else:
                        arr_ = [j for j in np.argsort(diffs) if diffs[j] / np.amax(diffs) > 0.4]
                    for spot_indx in arr_:
                        spot = spots[spot_indx]
                        x1, y1, w, h = spot

                        spot_crop = frame[y1:y1 + h, x1:x1 + w, :]

                        spot_status = empty_or_not(spot_crop)

                        spots_status[spot_indx] = spot_status

                if frame_nmr % step == 0:
                    previous_frame = frame.copy()

                for spot_indx, spot in enumerate(spots):
                    spot_status = spots_status[spot_indx]
                    x1, y1, w, h = spots[spot_indx]

                    if spot_status:
                        frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
                    else:
                        frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)

                cv2.rectangle(frame, (80, 20), (550, 80), (0, 0, 0), -1)
                cv2.putText(frame, 'Available spots: {} / {}'.format(str(sum(spots_status)), str(len(spots_status))), (100, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
                cv2.imshow('frame', frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

                frame_nmr += 1

            cap.release()
            cv2.destroyAllWindows()

            # Make sure to replace 'mask' and 'video_path' with the appropriate variables
            # and adjust the code as per your requirements.
            return "Files uploaded successfully and processing started."
    return render_template('upload.html')

@app.route('/')
def first_page():
    return render_template('first1.html')
@app.route('/first1')
def first1():
    return render_template('first1.html')
@app.route('/landing')
def landing():
    return render_template('landing.html')
if __name__ == '__main__':
    app.run(debug=True)
