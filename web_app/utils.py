from bert_serving.client import BertClient
import numpy as np
import pandas as pd
from scipy.spatial import distance
import requests
from bs4 import BeautifulSoup as bs
import os
import random

MEME_BERT = None
MEME_DF = None


def get_captioned_img(img_ids, save_path):
	save_names = []
	for img_id in img_ids:
		url = 'https://memegenerator.net/instance/' + str(img_id)
		r = requests.get(url)
		soup = bs(r.text, 'html.parser')
		image_url = soup.select_one('[rel="image_src"]')['href']
		save_name = os.path.join(save_path, f"{img_id}.jpg")
		response = requests.get(image_url, stream=True)

		with open(save_name, 'wb') as im_file:
			im_file.write(response.content)
		del response

		save_names.append(save_name)

	return save_names


def load_dialog_model():
	return None


def load_embedding_model():
	# Set up BERT server separately: bert-serving-start -model_dir /tmp/uncased_L-12_H-768_A-12 -num_worker=4
	global MEME_BERT
	MEME_BERT = np.load("/tmp/meme_bert.npz")
	MEME_BERT = MEME_BERT['arr_0']
	global MEME_DF
	MEME_DF = pd.read_csv("/tmp/meme_caption_processed.csv")


def get_reply(input_text):
	# TODO: import language models
	return " ".join(input_text)


def get_embedding(reply_text):
	bc = BertClient(ip='localhost')
	return bc.encode([reply_text])


def get_similar_meme(embedding):
	global MEME_BERT
	global MEME_DF
	if MEME_BERT is None:
		MEME_BERT = np.load("/tmp/meme_bert.npz")
		MEME_BERT = MEME_BERT['arr_0']
	if MEME_DF is None:
		MEME_DF = pd.read_csv("/tmp/meme_caption_processed.csv")

	sim_result = np.full((MEME_BERT.shape[0],), 0.0)
	for rdx in range(MEME_BERT.shape[0]):
		sim_result[rdx] = 1 - distance.cosine(embedding, MEME_BERT[rdx])
	best_match_meme_rdx = sim_result.argsort()[-3:]

	img_ids, captions, base_img_ids = [], [], []
	for rdx in best_match_meme_rdx:
		img_id, caption, base_img_id = MEME_DF.iloc[rdx]
		img_ids.append(img_id)
		captions.append(caption)
		base_img_ids.append(base_img_id)
	return img_ids, captions, base_img_ids


def get_meme(img_id, save_path):
	return get_captioned_img(img_id, save_path)


def get_random_meme():
	random_rx = random.randint(0, len(MEME_DF))
	img_id, caption, base_img_id = MEME_DF.iloc[random_rx]
	return img_id, caption, base_img_id


if __name__ == "__main__":
	pass
