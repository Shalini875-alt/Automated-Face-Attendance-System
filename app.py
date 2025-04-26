from flask import Flask, render_template, request, redirect, url_for
import face_recognition
import face_registration

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the Automated Face Attendance System!"

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # Call the face_registration function
        result = face_registration.register_face()
        return render_template('result.html', result=result)
    return '''
        <form method="POST">
            <button type="submit">Register Face</button>
        </form>
    '''

@app.route('/recognize', methods=['GET', 'POST'])
def recognize():
    if request.method == 'POST':
        # Call the face_recognition function
        result = face_recognition.recognize_face()
        return render_template('result.html', result=result)
    return '''
        <form method="POST">
            <button type="submit">Recognize Face</button>
        </form>
    '''

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

