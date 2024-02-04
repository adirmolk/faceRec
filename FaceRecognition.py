from flask import Flask, render_template, request
import cv2
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
from werkzeug.utils import secure_filename
app = Flask(__name__)

# Path to the folder where uploaded images will be stored
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# List of known face images along with corresponding names
known_faces = {
    "pete": cv2.imread("photos/pete.jpg"),
    "dicaprio": cv2.imread("photos/dicaprio.jpg"),
    "ariana": cv2.imread("photos/ariana.jpg"),
    "gal": cv2.imread("photos/gal.jpg"),

    # Add more images as needed
}

def compare_face_part(input_face_part, known_face):
    # Check if input_face_part and known_face are not None
    if input_face_part is None or known_face is None:
        print("Error: One of the input images is not loaded properly.")
        return 0  # Return a default similarity value or handle the error appropriately

    # Resize images to a common size
    input_face_part_resized = cv2.resize(input_face_part, (100, 100))
    known_face_resized = cv2.resize(known_face, (100, 100))

    # Convert images to grayscale
    gray_input_face_part = cv2.cvtColor(input_face_part_resized, cv2.COLOR_BGR2GRAY)
    gray_known_face = cv2.cvtColor(known_face_resized, cv2.COLOR_BGR2GRAY)

    # Calculate structural similarity index
    similarity, _ = ssim(gray_input_face_part, gray_known_face, full=True)

    return similarity



def recognize_person(input_face_part):
    # Dictionary to store similarity levels for each known person
    similarity_levels = {}

    if input_face_part is not None:
        for name, known_face in known_faces.items():
            similarity = compare_face_part(input_face_part, known_face)
            similarity_levels[name] = similarity

        # Find the person with the highest similarity
        best_match = max(similarity_levels, key=similarity_levels.get)
        max_similarity = similarity_levels[best_match]

        # Convert the similarity value to percentage
        similarity_percentage = int(max_similarity * 100)

        return best_match, similarity_percentage

    return None, 0


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', message='No file part')

        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', message='No selected file')

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            input_face_part = cv2.imread(file_path)

            if input_face_part is not None:
                result, similarity_percentage = recognize_person(input_face_part)
                return render_template('index.html', result=result, similarity_percentage=similarity_percentage, uploaded_file=filename)
            else:
                return render_template('index.html', message='Invalid file format. Please upload an image.')

    return render_template('index.html', message='Upload a picture')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png'}

if __name__ == '__main__':
    app.run(debug=True)