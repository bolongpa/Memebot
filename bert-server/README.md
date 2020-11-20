### Set up BERT Server Using TF 1.x

- Set up environment `venv_bert`
    - `virtualenv -p $(which python) ./venv_bert`
    - `pip install -r requirements.txt`
- Run BERT server locally
    - `bert-serving-start -model_dir /tmp/uncased_L-12_H-768_A-12 -num_worker=4`
    - Note that pretrained BERT model should be decompressed at `/tmp/uncased_L-12_H-768_A-12`
    - See more at https://github.com/hanxiao/bert-as-service/