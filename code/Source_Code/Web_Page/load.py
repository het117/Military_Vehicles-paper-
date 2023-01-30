from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import label_classific


app = Flask(__name__)

# 파일 업로드 처리
@app.route('/load')
def hellohtml():
    return render_template("upload.html")

@app.route('/load', methods=['GET', 'POST'])
def uploader_file():
    if request.method == 'POST':
        f = request.files['file']
        file_name = './file_upload/' + secure_filename(f.filename)
        f.save(file_name)
        
        result = label_classific.Result(file_name)
        
        return render_template('upload.html', re = result)

if __name__ == '__main__':
    # 서버 실행
    app.run(host = '0.0.0.0', port=8080, debug = True)