import  numpy as np
import sys
sys.path.append('...')
from model_helper import _conv_bn_relu
from model_helper import MergeLayers
from model_helper import PoolLayer
import keras
from keras.layers import (
  Input,
  Activation,
  Dense,
  Flatten,
  Dropout,
  Conv2D,
  MaxPooling2D,
  BatchNormalization,
  AveragePooling2D,
  GlobalAveragePooling2D,
  Concatenate)
from keras.models import Model
def BuildModel(input_shape, num_outputs, out_filters):
  ROW_AXIS = 1
  COL_AXIS = 2
  CHANNEL_AXIS = 3
  filters = out_filters
  input = Input(shape=input_shape)
  Layer_0 = Conv2D(filters=filters, kernel_size=(3,3), strides=(1, 1),padding="same",kernel_initializer="he_normal")(input)
  MergedLayer_0 = BatchNormalization(axis=CHANNEL_AXIS)(Layer_0)

  #2
  Layer_1 = _conv_bn_relu(filters=filters, kernel_size=(5,5), strides=(1, 1), padding="same")(MergedLayer_0)
  MergedLayer_1 =  MergeLayers(Layer_1, filters)

  #2 0
  Layer_2 = _conv_bn_relu(filters=filters, kernel_size=(5,5), strides=(1, 1), padding="same")(MergedLayer_1)
  MergedLayer_2 =  MergeLayers(Layer_2, filters)

  #3 0 1
  Layer_3 = _conv_bn_relu(filters=filters, kernel_size=(5,5), strides=(1, 1), padding="same")(MergedLayer_2)
  MergedLayer_3 =  MergeLayers(MergedLayer_2, Layer_3, filters)

  #1 0 0 1
  Layer_4 = _conv_bn_relu(filters=filters, kernel_size=(3,3), strides=(1, 1), padding="same")(MergedLayer_3)
  MergedLayer_4 =  MergeLayers(MergedLayer_3, Layer_4, filters)

  #0 1 0 0 0
  Layer_5 = _conv_bn_relu(filters=filters, kernel_size=(3,3), strides=(1, 1), padding="same")(MergedLayer_4)
  MergedLayer_5 =  MergeLayers(MergedLayer_1, Layer_5, filters)

  #1 1 1 0 0 1
  Layer_6 = _conv_bn_relu(filters=filters, kernel_size=(3,3), strides=(1, 1), padding="same")(MergedLayer_5)
  MergedLayer_6 =  MergeLayers(MergedLayer_1, MergedLayer_2, MergedLayer_5, Layer_6, filters)

  #0 1 0 0 0 0 0
  Layer_7 = _conv_bn_relu(filters=filters, kernel_size=(3,3), strides=(1, 1), padding="same")(MergedLayer_6)
  MergedLayer_7 =  MergeLayers(MergedLayer_1, Layer_7, filters)

  #3 0 0 0 0 0 0 0
  Layer_8 = _conv_bn_relu(filters=filters, kernel_size=(5,5), strides=(1, 1), padding="same")(MergedLayer_7)
  MergedLayer_8 =  MergeLayers(Layer_8, filters)

  PooledLayer_8 = PoolLayer(MergedLayer_8, filters)
  PooledLayer_7 = PoolLayer(MergedLayer_7, filters)
  PooledLayer_6 = PoolLayer(MergedLayer_6, filters)
  PooledLayer_5 = PoolLayer(MergedLayer_5, filters)
  PooledLayer_4 = PoolLayer(MergedLayer_4, filters)
  PooledLayer_3 = PoolLayer(MergedLayer_3, filters)
  PooledLayer_2 = PoolLayer(MergedLayer_2, filters)
  PooledLayer_1 = PoolLayer(MergedLayer_1, filters)
  filters *= 2

  #0 0 1 1 0 0 0 0 0
  Layer_9 = _conv_bn_relu(filters=filters, kernel_size=(3,3), strides=(1, 1), padding="same")(PooledLayer_8)
  MergedLayer_9 =  MergeLayers(PooledLayer_2, PooledLayer_3, Layer_9, filters)

  #3 1 1 1 1 0 0 0 0 0
  Layer_10 = _conv_bn_relu(filters=filters, kernel_size=(5,5), strides=(1, 1), padding="same")(MergedLayer_9)
  MergedLayer_10 =  MergeLayers(PooledLayer_1, PooledLayer_2, PooledLayer_3, PooledLayer_4, Layer_10, filters)

  #0 1 0 0 1 0 0 0 0 0 1
  Layer_11 = _conv_bn_relu(filters=filters, kernel_size=(3,3), strides=(1, 1), padding="same")(MergedLayer_10)
  MergedLayer_11 =  MergeLayers(PooledLayer_1, PooledLayer_4, MergedLayer_10, Layer_11, filters)

  #1 0 1 0 1 1 1 0 0 0 1 0
  Layer_12 = _conv_bn_relu(filters=filters, kernel_size=(3,3), strides=(1, 1), padding="same")(MergedLayer_11)
  MergedLayer_12 =  MergeLayers(PooledLayer_2, PooledLayer_4, PooledLayer_5, PooledLayer_6, MergedLayer_10, Layer_12, filters)

  #2 0 1 0 1 0 0 0 1 1 0 0 0
  Layer_13 = _conv_bn_relu(filters=filters, kernel_size=(5,5), strides=(1, 1), padding="same")(MergedLayer_12)
  MergedLayer_13 =  MergeLayers(PooledLayer_2, PooledLayer_4, PooledLayer_8, MergedLayer_9, Layer_13, filters)

  #3 0 1 1 1 0 0 1 0 0 0 0 0 0
  Layer_14 = _conv_bn_relu(filters=filters, kernel_size=(5,5), strides=(1, 1), padding="same")(MergedLayer_13)
  MergedLayer_14 =  MergeLayers(PooledLayer_2, PooledLayer_3, PooledLayer_4, PooledLayer_7, Layer_14, filters)

  #0 0 0 0 0 1 1 0 0 1 0 1 0 1 0
  Layer_15 = _conv_bn_relu(filters=filters, kernel_size=(3,3), strides=(1, 1), padding="same")(MergedLayer_14)
  MergedLayer_15 =  MergeLayers(PooledLayer_5, PooledLayer_6, MergedLayer_9, MergedLayer_11, MergedLayer_13, Layer_15, filters)

  #2 1 1 1 1 0 1 0 0 0 0 0 0 0 0 0
  Layer_16 = _conv_bn_relu(filters=filters, kernel_size=(5,5), strides=(1, 1), padding="same")(MergedLayer_15)
  MergedLayer_16 =  MergeLayers(PooledLayer_1, PooledLayer_2, PooledLayer_3, PooledLayer_4, PooledLayer_6, Layer_16, filters)

  S_PooledLayer_16 = PoolLayer(MergedLayer_16, filters)
  S_PooledLayer_15 = PoolLayer(MergedLayer_15, filters)
  S_PooledLayer_14 = PoolLayer(MergedLayer_14, filters)
  S_PooledLayer_13 = PoolLayer(MergedLayer_13, filters)
  S_PooledLayer_12 = PoolLayer(MergedLayer_12, filters)
  S_PooledLayer_11 = PoolLayer(MergedLayer_11, filters)
  S_PooledLayer_10 = PoolLayer(MergedLayer_10, filters)
  S_PooledLayer_9 = PoolLayer(MergedLayer_9, filters)
  S_PooledLayer_8 = PoolLayer(PooledLayer_8, filters)
  S_PooledLayer_7 = PoolLayer(PooledLayer_7, filters)
  S_PooledLayer_6 = PoolLayer(PooledLayer_6, filters)
  S_PooledLayer_5 = PoolLayer(PooledLayer_5, filters)
  S_PooledLayer_4 = PoolLayer(PooledLayer_4, filters)
  S_PooledLayer_3 = PoolLayer(PooledLayer_3, filters)
  S_PooledLayer_2 = PoolLayer(PooledLayer_2, filters)
  S_PooledLayer_1 = PoolLayer(PooledLayer_1, filters)
  filters *= 2

  #1 0 0 0 0 1 0 0 0 0 0 1 0 0 1 0 1
  Layer_17 = _conv_bn_relu(filters=filters, kernel_size=(3,3), strides=(1, 1), padding="same")(S_PooledLayer_16)
  MergedLayer_17 =  MergeLayers(S_PooledLayer_5, S_PooledLayer_11, S_PooledLayer_14, S_PooledLayer_16, Layer_17, filters)

  #0 1 1 1 1 1 1 0 1 0 0 0 0 0 0 0 0 0
  Layer_18 = _conv_bn_relu(filters=filters, kernel_size=(3,3), strides=(1, 1), padding="same")(MergedLayer_17)
  MergedLayer_18 =  MergeLayers(S_PooledLayer_1, S_PooledLayer_2, S_PooledLayer_3, S_PooledLayer_4, S_PooledLayer_5, S_PooledLayer_6, S_PooledLayer_8, Layer_18, filters)

  #0 0 0 0 1 0 0 0 1 0 1 1 0 0 0 1 0 0 0
  Layer_19 = _conv_bn_relu(filters=filters, kernel_size=(3,3), strides=(1, 1), padding="same")(MergedLayer_18)
  MergedLayer_19 =  MergeLayers(S_PooledLayer_4, S_PooledLayer_8, S_PooledLayer_10, S_PooledLayer_11, S_PooledLayer_15, Layer_19, filters)

  #3 0 0 1 0 1 0 0 0 1 1 1 1 1 1 0 0 0 0 1
  Layer_20 = _conv_bn_relu(filters=filters, kernel_size=(5,5), strides=(1, 1), padding="same")(MergedLayer_19)
  MergedLayer_20 =  MergeLayers(S_PooledLayer_3, S_PooledLayer_5, S_PooledLayer_9, S_PooledLayer_10, S_PooledLayer_11, S_PooledLayer_12, S_PooledLayer_13, S_PooledLayer_14, MergedLayer_19, Layer_20, filters)

  #2 1 0 0 1 0 0 0 1 1 1 0 0 1 1 1 1 0 1 0 1
  Layer_21 = _conv_bn_relu(filters=filters, kernel_size=(5,5), strides=(1, 1), padding="same")(MergedLayer_20)
  MergedLayer_21 =  MergeLayers(S_PooledLayer_1, S_PooledLayer_4, S_PooledLayer_8, S_PooledLayer_9, S_PooledLayer_10, S_PooledLayer_13, S_PooledLayer_14, S_PooledLayer_15, S_PooledLayer_16, MergedLayer_18, MergedLayer_20, Layer_21, filters)

  #3 0 0 1 0 0 0 0 0 1 0 0 0 0 1 0 0 1 1 0 1 1
  Layer_22 = _conv_bn_relu(filters=filters, kernel_size=(5,5), strides=(1, 1), padding="same")(MergedLayer_21)
  MergedLayer_22 =  MergeLayers(S_PooledLayer_3, S_PooledLayer_9, S_PooledLayer_14, MergedLayer_17, MergedLayer_18, MergedLayer_20, MergedLayer_21, Layer_22, filters)

  #2 0 0 1 1 0 1 0 1 0 0 1 0 1 0 0 0 1 1 0 0 0 1
  Layer_23 = _conv_bn_relu(filters=filters, kernel_size=(5,5), strides=(1, 1), padding="same")(MergedLayer_22)
  MergedLayer_23 =  MergeLayers(S_PooledLayer_3, S_PooledLayer_4, S_PooledLayer_6, S_PooledLayer_8, S_PooledLayer_11, S_PooledLayer_13, MergedLayer_17, MergedLayer_18, MergedLayer_22, Layer_23, filters)

  #3 1 0 0 0 1 0 0 0 1 0 1 1 0 0 0 0 1 1 1 0 1 1 0
  Layer_24 = _conv_bn_relu(filters=filters, kernel_size=(5,5), strides=(1, 1), padding="same")(MergedLayer_23)
  MergedLayer_24 =  MergeLayers(S_PooledLayer_1, S_PooledLayer_5, S_PooledLayer_9, S_PooledLayer_11, S_PooledLayer_12, MergedLayer_17, MergedLayer_18, MergedLayer_19, MergedLayer_21, MergedLayer_22, Layer_24, filters)

  FinalPooling = GlobalAveragePooling2D()(MergedLayer_24)
  dropout = Dropout(0.5)(FinalPooling)
  dense = Dense(units=num_outputs, kernel_initializer="he_normal", activation="softmax")(dropout)
  model = Model(inputs=input, outputs=dense)
  return model
