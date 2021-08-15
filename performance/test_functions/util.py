import warnings
warnings.filterwarnings("ignore")

import os
import csv

from os.path import join as join_path
import pandas as pd
abspath = os.path.dirname(os.path.abspath(__file__))


def filename_log(counter, x, y, z, a, b, n_layers, time, out_dir, trial):
	filename_dir = os.path.join(out_dir, "filename_log")
	if not os.path.exists(filename_dir):
		os.makedirs(filename_dir)
		print("============ created filename logs")
	file = os.path.join(filename_dir, "filenames_"+str(trial)+".csv")
	model_filename = x+"_"+str(counter)+"_"+str(y)+"_"+str(z)

	with open(file, 'a') as f:
		writer = csv.writer(f)
		writer.writerow([model_filename, x, y, z, a, b, n_layers, time])


def appendCSV(df, filename):

	if os.path.isfile(filename):

		with open(filename, "a") as csv:

			df.to_csv(csv, header = False, index = False)

	else:

		print(f"==========NO SUCH FILE {filename}, creating new file")

		df.to_csv(filename, index = False)


def get_image_size(model):
	img_sizes = {
					'xception': [299, 299],
					'inception_v3': [299, 299],
					'inceptionresnet_v2': [299, 299],
					'mobilenet': [224, 224],
					'vgg19': [224, 224]
				}
	img_size = img_sizes.get(model)
	img_size_3 = img_size.copy()
	img_size_3.extend([3])
	return img_size, img_size_3


def get_num_trainable(model):
	trainbale_layers = {
						'xception': [60],
						'inception_v3': [229, 280, 99999],
						'inceptionresnet_v2': [575, 758, 99999],
						'mobilenet': [103],
						'vgg19': [17]
						}

	# trainbale_layers = {
	# 					'xception': [56, 96, 126],
	# 					'inception_v3': [165, 229, 280],
	# 					'inceptionresnet_v2': [367, 575, 758],
	# 					'mobilenet': [73, 117, 144]
	# 					}		
	return trainbale_layers.get(model)


def get_weights_path(output_dir, pretrained, counter, trial):
	model_dir = os.path.join(output_dir, "models")
	# model_dir = f"E:/COVID/AI_CXR/transfer_learning/models_{data_distribution[0]}/"
	# model_dir = "model"
	return join_path(
					model_dir,
					"{}/{}_{}_{}.hdf5"
					.format(
					pretrained,
					pretrained,
					counter,
					trial
					)
					)