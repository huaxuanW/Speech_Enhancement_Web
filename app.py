from flask import Flask, render_template, request, session
from src.utils import noise_mixer, prediction
import os

app = Flask(__name__)

@app.route('/')
def home():
    print('restart')
    if 'clean_speech' in session:
        session.pop('clean_speech')
    if 'noise_speech' in session:
        session.pop('noise_speech')
        os.remove('static/temp/mixed_speech.wav')
    if 'enhanced_speech' in session:
        session.pop('enhanced_speech')
    return render_template('index.html')


@app.route("/upload", methods=["GET", "POST"])
def upload_file():
    if request.method=="POST":

        if "file" not in request.files: #如果文件不存在
            return home()
        
        file = request.files["file"]
        if file.filename == "":#如果文件名为空
            return home()
        
        filename = file.filename
        file.save('static/upload_file/' + filename)

        session['clean_speech'] = filename
        print(session['clean_speech'])
        return render_template('index.html', clean_speech=session['clean_speech'])


@app.route('/mix_<string:name>')
def mixer(name):
    if os.path.exists('static/temp/mixed_speech.wav'):
        print('delete')
        os.remove('static/temp/mixed_speech.wav')
        session.pop('mixed_speech') 

    print("noise type: ",name)
    if 'clean_speech' not in session:
        return home()

    noise_file = f'src/noise/{name}.pt'

    mixed_path = noise_mixer(session['clean_speech'], noise_file, snr=6, mixed_path='mixed_speech.wav')

    session['mixed_speech'] = mixed_path
    session['noise_speech'] = "temp_noise.wav"
    print(mixed_path)
    return render_template('index.html', clean_speech=session['clean_speech'], mixed_speech=session['mixed_speech'])


@app.route('/enhanced')
def removeNoise():
    if 'clean_speech' not in session:
        return home()
    if 'mixed_speech' not in session:
        return home()
    prediction(session['clean_speech'], session['noise_speech'], session['mixed_speech'], 'enhanced_speech.wav')
    session['enhanced_speech'] = 'enhanced_speech.wav'

    return render_template('index.html', clean_speech=session['clean_speech'], mixed_speech=session['mixed_speech'], enhanced_speech=session['enhanced_speech'])

@app.route('/home')
def reset():
    return home()


if __name__ == "__main__":
    app.secret_key='huaxuan'
    app.run(debug=True, threaded= True)