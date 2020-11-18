import os

from flask import Flask, render_template, request

from web_app.ocr_core import ocr_core

from web_app.utils import *

# TODO:import memebot model method


UPLOAD_FOLDER = './static/uploads/'
DOWNLOAD_FOLDER = './static/downloads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

# Safety check
if not os.path.isdir('./static'):
    os.mkdir('./static')

if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)

if not os.path.isdir(DOWNLOAD_FOLDER):
    os.mkdir(DOWNLOAD_FOLDER)


app = Flask(__name__, static_folder='./static')

# 用model模型数据代替“_”，有多个数据（比如对话model可能需要embedding model和encoder-decoder model）的依次列出
# _ = load_similarity_model()

_ = load_dialog_model()

# 不知道embedding需不需要单独的模型，不需要的话就删掉这一步load然后把embedding部分并入到get_picture函数就行了
_ = load_embedding_model()


def allowed_file(filename):
    return '.' in filename.lower() and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home_page():
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload_page():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return render_template('upload.html', msg='No file selected')
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            return render_template('upload.html', msg='No file selected')

        if file and allowed_file(file.filename):
            file.save(os.path.join(UPLOAD_FOLDER, file.filename))

            # call the OCR function on it
            # 第一步，用OCR提取图片里的text？
            input_text = ocr_core(file)
            reply_meme_dir = reply_meme(input_text)
            print("ATTENTION", input_text, reply_meme_dir)

            # extract the text and display it
            return render_template('upload.html',
                                   msg='Successfully processed',
                                   reply_meme=reply_meme_dir,
                                   img_src=UPLOAD_FOLDER + file.filename)
    elif request.method == 'GET':
        return render_template('upload.html')


# 返回一个生成好的图片的网址
def reply_meme(input_text):
    # 第二步，把input_text送进对话模型算出reply_text
    # 把_替换成对话模型数据
    # reply_text = get_reply(_, input_text)
    reply_text = "how are you"
    # 第三步， 把reply_text送进embedding model算embedding
    # 把_替换成embedding模型数据
    reply_embedding = get_embedding(reply_text)
    # 第四步，把embedding送进bert模型算出匹配的meme图片
    # 把_替换成bert模型数据
    img_id, caption, base_img_id = get_similar_meme(reply_embedding)
    # 第五步，把图片和原本的reply一起生成新的meme并返回meme链接
    meme_save_path = get_meme(img_id, DOWNLOAD_FOLDER)
    return meme_save_path


if __name__ == '__main__':
    app.run(debug=True)
    # reply_text = "how are you"
    # reply_embedding = get_embedding(reply_text)
    # img_id, caption, base_img_id = get_similar_meme(reply_embedding)
    # meme_save_path = get_meme(img_id, DOWNLOAD_FOLDER)
    # print(img_id, caption, base_img_id)
