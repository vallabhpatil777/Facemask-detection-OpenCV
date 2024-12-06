from flask import Flask, render_template, Response
import cv2
import numpy as np
import pickle


app = Flask(__name__)

with open('model2.pkl', 'rb') as f:
    model = pickle.load(f)

camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            print("Error: Failed to read frame from camera.")
            break
        else:
            frame = cv2.flip(frame,1)
            img = cv2.resize(frame, (224, 224))
            img = img / 255.0
            
            y_pred = detect_face_mask(img)

            if y_pred == 0:
                draw_label(frame, "Mask", (30, 30), (0, 255, 0))  
            else:
                draw_label(frame, "No Mask", (30, 30), (0, 0, 255))  
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def detect_face_mask(frame):
    
    prediction = model.predict(frame.reshape(1, 224, 224, 3))[0][0]
    print(prediction)
    
    return 0 if prediction < 0.5 else 1  

def draw_label(img, text, pos, bg_color):
    text = str(text)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 4  # Adjusted font scale for better readability
    thickness = 2  # Font thickness
    padding = 5  # Padding around the text

    # Get the text size
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_width, text_height = text_size

    # Calculate rectangle coordinates
    end_x = pos[0] + text_width + 2 * padding
    end_y = pos[1] + text_height + 2 * padding

    # Draw the background rectangle
    cv2.rectangle(img, pos, (end_x, end_y), bg_color, -1)

    # Draw the text on top of the rectangle
    text_x = pos[0] + padding
    text_y = pos[1] + text_height + padding
    cv2.putText(img, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
