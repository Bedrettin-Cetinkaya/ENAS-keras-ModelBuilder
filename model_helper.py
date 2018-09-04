import  numpy as np
import keras
from keras.layers import (
  Input,
  Activation,
  Dense,
  Flatten,
  Dropout,
  Conv2D,
  MaxPooling2D,
  AveragePooling2D,
  GlobalAveragePooling2D,
  BatchNormalization,
  Concatenate)


global ROW_AXIS
global COL_AXIS
global CHANNEL_AXIS
ROW_AXIS = 1
COL_AXIS = 2
CHANNEL_AXIS = 3

def _conv_bn_relu(**conv_params):
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    def f(input):
        relu1 = Activation("relu")(input)
        conv1_1 = Conv2D(filters=filters, kernel_size=(1,1),
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer)(relu1)
        norm1 = BatchNormalization(axis=CHANNEL_AXIS)(conv1_1)

        relu2 = Activation("relu")(norm1)
        convSpec = Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer)(relu2)
        norm2 = BatchNormalization(axis=CHANNEL_AXIS)(convSpec)
        return norm2

    return f

def DoConv1_1(input,filters):
  relu = Activation("relu")(input)
  conv1_1 = Conv2D(filters=filters, kernel_size=(1,1),
                   strides=(1,1),
                   padding="same",
                   kernel_initializer="he_normal")(relu)
  norm = BatchNormalization(axis=CHANNEL_AXIS)(conv1_1)
  return norm

def MergeLayer(Input1,Input2):
  return Concatenate(axis=3)([Input1,Input2])

def MergeLayers(*Arg):
  MergedLayers = Arg[0]
  for i in range(1,len(Arg) -1):
    MergedLayers = MergeLayer(MergedLayers,Arg[i])
  return DoConv1_1(MergedLayers, Arg[-1])

def PoolLayer(Layer,filters):
  P1_Layer = AveragePooling2D(pool_size=(1, 1), strides=(2,2), padding='valid')(Layer)
  P1_Conv1_1 = Conv2D(filters=filters, kernel_size=(1,1),
                      strides=(1,1), padding="same",
                      kernel_initializer="he_normal")(P1_Layer)
  P2_Layer = AveragePooling2D(pool_size=(1, 1), strides=(2,2), padding='valid')(Layer)
  P2_Conv1_1 = Conv2D(filters=filters, kernel_size=(1,1),
                      strides=(1,1), padding="same",
                      kernel_initializer="he_normal")(P2_Layer) 
  merge = MergeLayer(P1_Conv1_1,P2_Conv1_1)
  Norm = BatchNormalization(axis=CHANNEL_AXIS)(merge)
  return Norm
