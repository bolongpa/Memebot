[![HitCount](http://hits.dwyl.io/ro6ley/python-ocr-example.svg)](http://hits.dwyl.io/ro6ley/python-ocr-example)

### Running the Application

1. Install the requirements
```
$ virtualenv -p $(which python) ./venv_web_app
$ source venv_web_app/bin/activate
$ pip install -r requirements.txt
```

2. Run server
```
$ python app.py
```

## TODO

- replace ocr_core.py with model of meme bot. Clear /static/reply/ and put reply meme into this directory.
- meme bot python snippet takes image, and output reply directory in /static/reply/ 
