from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import os
import subprocess
from googletrans import Translator

app = Flask(__name__)
CORS(app)  # To allow cross-origin requests

UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

translator = Translator()

@app.route('/generate-description', methods=['POST'])
def generate_description():
    print('get image')
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image = request.files['image']
    print('got image')
    if image.filename == '':
        return jsonify({'error': 'No selected image'}), 400
    
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
    image.save(image_path)  # Save the uploaded image
    
    try:
        # Run the image description generation script
        print('run image_description')
        env = os.environ.copy()
        env['PYTHONPATH'] = f"{env.get('PYTHONPATH', '')}:/home/yashraj/tts_image_description_app/models/Expansion_new/ExpansionNet_v2"
        result = subprocess.run([
            'python3',
            '/home/yashraj/tts_image_description_app/backend/img_des.py',
            '--load_path', '/home/yashraj/tts_image_description_app/models/Expansion_new/model_files/rf_model.pth',
            '--image_paths', image_path
        ], capture_output=True,env=env, text=True)
        
        if result.returncode != 0:
            return jsonify({'error': 'Error generating description', 'details': result.stderr}), 500
        
        # Get the description from stdout
        image_description = result.stdout.strip()
        
        # Detect the language or default to 'Marathi'
        lang = request.form.get('lang', 'Marathi')

        # Translate description to selected language using Google Translate
        translated_text = translator.translate(image_description, dest=lang).text
        
        # Run the TTS model to generate the audio
        env = os.environ.copy()
        tts_command = [
            'conda', 'run', '-n', 'tts-mfa-hifigan', 'python3', 'inference.py',
            '--sample_text', translated_text,
            '--language', lang,
            '--gender', 'male',
            # '--output_file', 'output.wav'
        ]
        env['PYTHONPATH'] = f"{env.get('PYTHONPATH', '')}:/home/yashraj/tts_image_description_app/models/Fastspeech2_MFA"
        tts_result = subprocess.run(tts_command, cwd='/home/yashraj/tts_image_description_app/models/Fastspeech2_MFA',env = env, capture_output=True, text=True)
        
        if tts_result.returncode != 0:
            print(tts_result)
            return jsonify({'error': 'Error generating TTS', 'details': tts_result.stderr}), 500
        
        output_wav_path = os.path.join('/home/yashraj/tts_image_description_app/backend', 'output.wav')
        
        if os.path.exists(output_wav_path):
            return send_file(output_wav_path, mimetype='audio/wav')
        else:
            return jsonify({'error': 'WAV file not found'}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=3000, debug=True)
