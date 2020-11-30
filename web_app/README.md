[![HitCount](http://hits.dwyl.io/ro6ley/python-ocr-example.svg)](http://hits.dwyl.io/ro6ley/python-ocr-example)

### Running the Application

- Install the requirements
```
$ virtualenv -p $(which python) ./venv_web_app
$ source venv_web_app/bin/activate
$ pip install -r requirements.txt
```
- Install Tesseract using `https://tesseract-ocr.github.io/tessdoc/Compiling.html`
- Download needed files from `https://drive.google.com/drive/folders/1w84PaKTIBzR4lSEF_h9LdVn-BPl7QgYl?usp=sharing`
    - `tokenizer.pickle` -> `/tmp/tokenizer.pickle`
    - `saved_weights.h5` -> `/tmp/saved_weights.h5`
    - `meme_bert.npz` -> `/tmp/meme_bert.npz`
    - `meme_caption_processed.csv` -> `/tmp/meme_caption_processed.csv`
- Run server
```
$ python app.py
```
- Visit `localhost:5000/upload` for meme chat
- Visit `localhost:5000/evaluation` for human evaluation
