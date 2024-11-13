from flask import Flask, request, jsonify
import os
app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    ocr_extracted_text = extract_text(filepath)
    emails=extract_emails(ocr_extracted_text)
    phone=extract_phone_numbers(ocr_extracted_text)
    nlp_entities = extract_entities(ocr_extracted_text)
    return jsonify({"emails":emails,"phones":phone,"ocr_text":ocr_extracted_text,"nlp":nlp_entities}),200

if __name__ == '__main__':
    app.run(debug=True)
