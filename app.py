import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from flask import send_from_directory
import librosa
import matplotlib.pyplot as plt
import numpy as np
import librosa.display

UPLOAD_FOLDER = 'app/static/'
ALLOWED_EXTENSIONS = {'mp3', 'wav'}

app = Flask(__name__, template_folder='templates/')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('draw_spectr',
                                    filename=filename))
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['IMAGE_STORE_PATH'],
                               filename)


@app.route('/audio_features/<filename>')
def draw_spectr(filename):
    y, sr = librosa.load(f'app/static/{filename}')
    D = np.abs(librosa.stft(y))
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), y_axis='log', x_axis='time')
    plt.title('Power spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(f'static/{filename}.png')
    return render_template('spectrogramm.html', filename=f'{filename}.png', audioname=filename)


@app.route('/all_features/<filename>')
def extract_features(filename):
    y, sr = librosa.load(f'app/static/{filename}')
    plt.figure(figsize=(14, 5))
    librosa.display.waveplot(y, sr=sr)
    plt.savefig('static/waveplot.png')
    mfcc = librosa.feature.mfcc(y, sr=sr)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(mfcc, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.savefig('static/MFCC.png')
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(mfcc_delta, x_axis='time')
    plt.colorbar()
    plt.title(r'MFCC-$\Delta$')
    plt.savefig('static/MFCC_delta.png')
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(mfcc_delta2, x_axis='time')
    plt.colorbar()
    plt.title(r'MFCC-$\Delta^2$')
    plt.savefig('static/MFCC_delta2.png')
    S, phase = librosa.magphase(librosa.stft(y))
    rms = librosa.feature.rms(S=S)
    plt.figure(figsize=(14, 5))
    plt.semilogy(rms.T, label='RMS Energy')
    plt.xticks([])
    plt.xlim([0, rms.shape[-1]])
    plt.legend()
    plt.savefig('static/RMS.png')
    return render_template('all_features.html', waveplot='waveplot.png', mfcc='MFCC.png', mfcc_delta='MFCC_delta.png',
                           mfcc_delta2='MFCC_delta2.png', rms='rms.png')
