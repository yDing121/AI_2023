# import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import ELU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K
# 网络构建
class EmotionVGGNet:
	@staticmethod
	# 指定网络输入的大小和输出
	def build(width, height, depth, classes):
		# initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
		# 序列方式
		model = Sequential()
		# 输入
		inputShape = (height, width, depth)
		chanDim = -1

		# if we are using "channels first", update the input shape
		# and channels dimension
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1
		# 构建网络：以pool将网络分割成多个模块
		# Block #1: first CONV => RELU => CONV => RELU => POOL
		# layer set
		model.add(Conv2D(32, (3, 3), padding="same",
			kernel_initializer="he_normal", input_shape=inputShape))
		model.add(ELU())
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(32, (3, 3), kernel_initializer="he_normal",
			padding="same"))
		model.add(ELU())
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		# Block #2: second CONV => RELU => CONV => RELU => POOL
		# layer set
		model.add(Conv2D(64, (3, 3), kernel_initializer="he_normal",
			padding="same"))
		model.add(ELU())
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(64, (3, 3), kernel_initializer="he_normal",
			padding="same"))
		model.add(ELU())
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		# Block #3: third CONV => RELU => CONV => RELU => POOL
		# layer set
		model.add(Conv2D(128, (3, 3), kernel_initializer="he_normal",
			padding="same"))
		model.add(ELU())
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(128, (3, 3), kernel_initializer="he_normal",
			padding="same"))
		model.add(ELU())
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		# Block #4: first set of FC => RELU layers
		model.add(Flatten())
		model.add(Dense(64, kernel_initializer="he_normal"))
		model.add(ELU())
		model.add(BatchNormalization())
		model.add(Dropout(0.5))

		# Block #6: second set of FC => RELU layers
		model.add(Dense(64, kernel_initializer="he_normal"))
		model.add(ELU())
		model.add(BatchNormalization())
		model.add(Dropout(0.5))

		# Block #7: softmax classifier
		model.add(Dense(classes, kernel_initializer="he_normal"))
		model.add(Activation("softmax"))

		# return the constructed network architecture
		return model