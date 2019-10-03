from flask import Flask, render_template, request, redirect

import os
import Last_Try_Test
import shutil

input_dir = './static/test/input/pic'
app = Flask(__name__,static_folder='./static')
app.config['UPLOAD_FOLDER'] = input_dir
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1




# Route for home or index
@app.route('/')
def home():
    shutil.rmtree('./static/test/input/pic')
    os.mkdir('./static/test/input/pic')
    return render_template('home.html')


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
        Last_Try_Test.test()
    return redirect('http://localhost:5000/test')


@app.route('/test', methods=['GET'])
def test():
    return render_template('test.html')

@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response


if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=5000)
