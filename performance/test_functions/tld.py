import warnings
warnings.filterwarnings("ignore")

import os
import time
import pathlib
import numpy as np
import pandas as pd

from itertools import compress
import tensorflow as tf

from tensorflow.keras import regularizers
from tensorflow.keras import Model, Input
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg19 import preprocess_input

import itertools

import util
import metrics
import models

AUTOTUNE = tf.data.experimental.AUTOTUNE

def create_dataset():

	img_size, _ = util.get_image_size("inception_v3")
	
	data_dir = pathlib.Path(base_dir)

	data_image_count = len(list(data_dir.glob('*/*')))

	global class_names, class_number, name_id_map

	class_names = np.array([item.name for item in data_dir.glob('*') \
							if not item.name.endswith(".DS_Store")])

	
	name_id_map = dict(zip(range(len(class_names)), class_names))

	class_number = len(class_names)

	data_list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'))

	data_steps = np.ceil(data_image_count/16)

	def get_label(file_path):
		parts = tf.strings.split(file_path, os.path.sep)
		return parts[-2] == class_names

	def decode_img(img):
		img = tf.io.decode_image(img, channels=3, expand_animations=False)
		img = tf.image.convert_image_dtype(img, tf.float32)
		return tf.image.resize(img, img_size, method=tf.image.ResizeMethod.BICUBIC)

	def process_path(file_path):
		label = get_label(file_path)
		img = tf.io.read_file(file_path)
		img = decode_img(img)
		return img, label

	data_list_ds = data_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

	data_batches = (
						data_list_ds
						.cache()
						.batch(16)
						.prefetch(AUTOTUNE)
					) 

	labels = [labels for images, labels in data_batches.take(-1)]
	images = [images for images, labels in data_batches.take(-1)]

	all_label = []
	for label in labels:
		all_label.extend(label.numpy())

	all_label = np.array(all_label)

	y_true = np.concatenate([np.where(i)[0] for i in all_label], axis = 0)

	return y_true, images


def tld_create_model(ver):

	base_model, pred = models.initialize_inceptionv3(
												512, 
												regularizers.l2(0.001),
												0.2, 
												2
							)

	model_1 = Model(inputs=base_model.input, outputs=pred)
	opt = tf.keras.optimizers.SGD(1e-5, momentum = 0.9)

	model_1.load_weights(f"model/{ver}/pneumonia_detector.hdf5")

	model_1.compile(
					loss = "categorical_crossentropy",
					optimizer = opt,
					metrics = ["accuracy", 
								metrics.ppv, 
								metrics.npv
								]
				)
	
	base_model, pred = models.initialize_vgg19(
												512, 
												regularizers.l2(0.001),
												0.3, 
												3
							)

	model_2 = Model(inputs=base_model.input, outputs=pred)
	opt = tf.keras.optimizers.SGD(1e-5, momentum = 0.9)

	model_2.load_weights(f"model/{ver}/covid_detector.hdf5")

	model_2.compile(
					loss = "categorical_crossentropy",
					optimizer = opt,
					metrics = ["accuracy", 
								metrics.ppv, 
								metrics.npv
								]
				)

	return model_1, model_2

def compare_predictions(x):
	if x["model1"] == 0:
		return x["model1"]
	else:
		return x["model2"]


def tld_predict(ver, images):
	model_1, model_2 = create_model(ver)
	y_pred_1 = []
	for image in images:
		preds = model_1.predict(image)
		y_pred_1.extend(np.argmax(preds, axis = -1))

	y_pred_2 = []
	for image in images:
		preds = model_2.predict(image)
		y_pred_2.extend(np.argmax(preds, axis = -1))

	compare_pred = pd.DataFrame(data = y_pred_1, columns = ["model1"])
	compare_pred["model2"] = y_pred_2
	compare_pred["model2"] = compare_pred["model2"] + 1
	compare_pred["final"] = compare_pred.apply(compare_predictions, axis = 1)

	return compare_pred["final"]


def four_class_create():
	base_model, pred = models.initialize_inceptionv3(
												512, 
												regularizers.l2(0.001),
												0.2, 
												4
							)

	model = Model(inputs=base_model.input, outputs=pred)
	opt = tf.keras.optimizers.SGD(1e-5, momentum = 0.9)

	model.load_weights(f"model/four_class/four_class.hdf5")

	model.compile(
					loss = "categorical_crossentropy",
					optimizer = opt,
					metrics = ["accuracy", 
								metrics.ppv, 
								metrics.npv
								]
				)
	return model


def four_class_predict(ver, images):
	model = create_model(ver)
	y_pred = []
	for image in images:
		preds = model.predict(image)
		y_pred.extend(np.argmax(preds, axis = -1))

	return y_pred


def create_model(ver):
	if "tld" in ver:
		return tld_create_model(ver)
	elif "four" in ver:
		return four_class_create()
	else:
		display("Version not available")


def get_predictions(ver):
	y_true, images = create_dataset()
	if "tld" in ver:
		y_pred = tld_predict(ver, images)
	elif "four" in ver:
		y_pred = four_class_predict(ver, images)
	else:
		display("Version not available")

	return y_true, y_pred



def main(image_dir, ver):
	global base_dir
	base_dir = image_dir

	y_true, y_pred = get_predictions(ver)
	metrics.get_metrics(y_true, y_pred, class_names, name_id_map)
	metrics.plot_confmatrix(y_true, y_pred)



if __name__ == '__main__':
	main()
