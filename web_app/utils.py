from bert_serving.client import BertClient
import numpy as np
import pandas as pd
from scipy.spatial import distance
import requests
from bs4 import BeautifulSoup as bs
import os
import random
# for dialog model
import tensorflow as tf
import pickle
from transformer_model import textPreprocess
from transformer_model import transformer
from transformer_model import CustomSchedule
from transformer_model import accuracy, loss_function


MEME_BERT = None
MEME_DF = None

DIALOG_TRANS = None
DIALOG_TOKEN = None

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
	strategy = tf.distribute.get_strategy()

	MAX_LENGTH = 30
	BATCH_SIZE = int(64 * strategy.num_replicas_in_sync)
	BUFFER_SIZE = 20000

	NUM_LAYERS = 2 #6
	D_MODEL = 256 #512
	NUM_HEADS = 8
	UNITS = 512 #2048
	DROPOUT = 0.1

	EPOCHS = 100

	global DIALOG_TOKEN
	with open('tokenizer.pickle', 'rb') as handle:
		DIALOG_TOKEN = pickle.load(handle)

	START_TOKEN, END_TOKEN = [DIALOG_TOKEN.vocab_size], [DIALOG_TOKEN.vocab_size + 1]
	VOCAB_SIZE = DIALOG_TOKEN.vocab_size + 2

	learning_rate = CustomSchedule(D_MODEL)
	optimizer = tf.keras.optimizers.Adam(
		learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

	global DIALOG_TRANS
	DIALOG_TRANS = transformer(
		vocab_size=VOCAB_SIZE,
		num_layers=NUM_LAYERS,
		units=UNITS,
		d_model=D_MODEL,
		num_heads=NUM_HEADS,
		dropout=DROPOUT)
	DIALOG_TRANS.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy])
	DIALOG_TRANS.load_weights('saved_weights.h5')



def load_embedding_model():
	# Set up BERT server separately: bert-serving-start -model_dir /tmp/uncased_L-12_H-768_A-12 -num_worker=4
	global MEME_BERT
	MEME_BERT = np.load("/tmp/meme_bert.npz")
	MEME_BERT = MEME_BERT['arr_0']
	global MEME_DF
	MEME_DF = pd.read_csv("/tmp/meme_caption_processed.csv")


def get_reply(input_text):
	global DIALOG_TOKEN
	global DIALOG_TRANS

	def evaluate(sentence, model):
		sentence = textPreprocess(sentence)
		sentence = tf.expand_dims(
			START_TOKEN + DIALOG_TOKEN.encode(sentence) + END_TOKEN, axis=0)
		output = tf.expand_dims(START_TOKEN, 0)

		for i in range(MAX_LENGTH):
		predictions = model(inputs=[sentence, output], training=False)
		
		predictions = predictions[:, -1:, :]
		predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
		
		if tf.equal(predicted_id, END_TOKEN[0]):
			break
		
		output = tf.concat([output, predicted_id], axis=-1)

		return tf.squeeze(output, axis=0)

	sentence = " ".join(input_text).strip(' ')
	prediction = evaluate(sentence, DIALOG_TRANS)
	predicted_sentence = DIALOG_TOKEN.decode(
	[i for i in prediction if i < DIALOG_TOKEN.vocab_size])
	
	return predicted_sentence.lstrip()


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
