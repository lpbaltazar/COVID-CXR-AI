import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import Model
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras import regularizers


def initialize_xception(FC_SIZE, REG, DROPOUT, class_number):
	base_model = Xception(weights='imagenet', include_top=False)
	x = base_model.output
	x = GlobalAveragePooling2D()(x)

	x = Dense(512, activation='relu', kernel_regularizer=REG)(x)
	x = Dropout(DROPOUT)(x)
	x = Dense(256, activation='relu', kernel_regularizer=REG)(x)
	x = Dropout(DROPOUT)(x)

	embedding = Dense(FC_SIZE, activation='relu', 
						name="embedding_layer", kernel_regularizer=REG)(x)
	predictions = Dense(class_number, activation='softmax',
						 name="prediction_layer", kernel_regularizer=REG, 
						 dtype='float32')(embedding)
	return base_model, predictions


def initialize_inceptionv3(FC_SIZE, REG, DROPOUT, class_number):
	base_model = InceptionV3(weights='imagenet', include_top=False)

	x = base_model.output
	x = GlobalAveragePooling2D()(x)

	x = Dense(512, activation='relu', kernel_regularizer=REG)(x)
	x = Dropout(DROPOUT)(x)
	x = Dense(256, activation='relu', kernel_regularizer=REG)(x)
	x = Dropout(DROPOUT)(x)

	embedding = Dense(FC_SIZE, activation='relu', 
						name="embedding_layer", kernel_regularizer=REG)(x)

	predictions = Dense(class_number, activation='softmax', 
						name="prediction_layer", kernel_regularizer=REG, 
						dtype='float32')(embedding)
	return base_model, predictions


def initialize_inceptionresnetv2(FC_SIZE, REG, DROPOUT, class_number):
	base_model = InceptionResNetV2(weights='imagenet', include_top=False)

	x = base_model.output
	x = GlobalAveragePooling2D()(x)

	x = Dense(512, activation='relu', kernel_regularizer=REG)(x)
	x = Dropout(DROPOUT)(x)
	x = Dense(256, activation='relu', kernel_regularizer=REG)(x)
	x = Dropout(DROPOUT)(x)

	embedding = Dense(FC_SIZE, activation='relu', 
						name="embedding_layer", kernel_regularizer=REG)(x)

	predictions = Dense(class_number, activation='softmax', 
						name="prediction_layer", kernel_regularizer=REG, 
						dtype='float32')(embedding)
	return base_model, predictions


def initialize_mobilenet(FC_SIZE, REG, DROPOUT, class_number):
	base_model = MobileNetV2(weights='imagenet', include_top=False)

	x = base_model.output
	x = GlobalAveragePooling2D()(x)

	x = Dense(512, activation='relu', kernel_regularizer=REG)(x)
	x = Dropout(DROPOUT)(x)
	x = Dense(256, activation='relu', kernel_regularizer=REG)(x)
	x = Dropout(DROPOUT)(x)

	embedding = Dense(FC_SIZE, activation='relu', 
						name="embedding_layer", kernel_regularizer=REG)(x)

	predictions = Dense(class_number, activation='softmax', 
						name="prediction_layer", kernel_regularizer=REG, 
						dtype='float32')(embedding)
	return base_model, predictions


def initialize_vgg19(FC_SIZE, REG, DROPOUT, class_number):
	base_model = VGG19(weights='imagenet', include_top=False)

	x = base_model.output
	x = GlobalAveragePooling2D()(x)

	x = Dense(512, activation='relu', kernel_regularizer=REG)(x)
	x = Dropout(DROPOUT)(x)
	x = Dense(256, activation='relu', kernel_regularizer=REG)(x)
	x = Dropout(DROPOUT)(x)

	embedding = Dense(FC_SIZE, activation='relu', 
						name="embedding_layer", kernel_regularizer=REG)(x)

	predictions = Dense(class_number, activation='softmax', 
						name="prediction_layer", kernel_regularizer=REG, 
						dtype='float32')(embedding)
	return base_model, predictions