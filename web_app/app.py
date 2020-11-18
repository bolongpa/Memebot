from flask import Flask, render_template, request
from web_app.ocr_core import ocr_core
from web_app.utils import *
from PIL import Image
import imagehash

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

_ = load_dialog_model()
_ = load_embedding_model()


@app.route('/')
def home_page():
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload_page():
    if request.method == 'POST':
        # Form submission check: if form is not empty, append to out.txt
        current_form = request.form.to_dict()
        if bool(current_form):
            chosen_key = list(current_form.keys())[0]
            chosen_val = current_form[chosen_key]
            image_hash = imagehash.average_hash(Image.open(chosen_val))
            with open("out.txt", "a+") as f:
                f.write(f"{image_hash} $ {chosen_key}")
                f.write("\n")
            print("Successfully appended to out.txt!")
            return render_template('upload.html', msg='No file selected')
        # Safety check for user upload
        if 'file' not in request.files:
            return render_template('upload.html', msg='No file selected')
        file = request.files['file']
        if file.filename == '':
            return render_template('upload.html', msg='No file selected')

        if file and allowed_file(file.filename):
            saved_file_name = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(saved_file_name)

            # Extract meme caption from input image
            input_text = ocr_core(saved_file_name)
            meme_save_paths = reply_meme(input_text)
            random_memes = random_meme()

            # Display generated meme
            # TODO: display randomized meme for evaluation
            return render_template('upload.html',
                                   msg='Successfully processed',
                                   best_reply_meme=meme_save_paths[0],
                                   second_reply_meme=meme_save_paths[1],
                                   third_reply_meme=meme_save_paths[2],
                                   random_meme=random_memes[0],
                                   img_src=UPLOAD_FOLDER + file.filename)
    elif request.method == 'GET':
        return render_template('upload.html')


def reply_meme(input_text):
    # Generate text reply using language model
    reply_text = get_reply(input_text)

    # TODO: currently reply_text can not be empty
    # Generate BERT embedding
    if len(reply_text.strip()) == 0:
        reply_text = "what's up bro"
    reply_embedding = get_embedding(reply_text)

    # Find nearest top k=3 neighbors to choose from
    img_ids, captions, base_img_ids = get_similar_meme(reply_embedding)
    meme_save_paths = get_meme(img_ids, DOWNLOAD_FOLDER)
    return meme_save_paths


def random_meme():
    img_id, caption, base_img_id = get_random_meme()
    meme_save_paths = get_meme([img_id], DOWNLOAD_FOLDER)
    print(img_id, meme_save_paths)
    return meme_save_paths


def allowed_file(filename):
    return '.' in filename.lower() and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


if __name__ == '__main__':
    # run server in web_app/: python app.py
    app.run(debug=True)

    # reply_text = "what's up bro"
    #
    # # Generate BERT embedding
    # reply_embedding = get_embedding(reply_text)
    #
    # # Find nearest top k=1 neighbors to choose from
    # img_ids, captions, base_img_ids = get_similar_meme(reply_embedding)
    # meme_save_paths = get_meme(img_ids, DOWNLOAD_FOLDER)
    # print(len(img_ids))
