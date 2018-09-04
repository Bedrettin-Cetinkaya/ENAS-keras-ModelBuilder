import numpy as np
import os.path
from writerHelper import MergeLayerstringBuilder
from writerHelper import PooledLayerStringBuilder

def DefaultCodesToFile(OutputDir):
  with open( OutputDir + '/Model.py', 'a') as f:
    f.write("import  numpy as np\n")
    f.write("import sys\n")
    f.write("sys.path.append('...')\n")
    f.write("from model_helper import _conv_bn_relu\n")
    f.write("from model_helper import MergeLayers\n")
    f.write("from model_helper import PoolLayer\n")
    f.write("import keras\n")
    f.write("from keras.layers import (\n  Input,\n  Activation,\n  Dense,\n  Flatten,\n  Dropout,\n  Conv2D,\n  MaxPooling2D,\n  BatchNormalization,\n  AveragePooling2D,\n")
    f.write("  GlobalAveragePooling2D,\n  Concatenate)\n")
    f.write("from keras.models import Model\n")
    f.write("def BuildModel(input_shape, num_outputs, out_filters):\n")
    f.write("  ROW_AXIS = 1\n")
    f.write("  COL_AXIS = 2\n")
    f.write("  CHANNEL_AXIS = 3\n")
    f.write("  filters = out_filters\n")
    f.write("  input = Input(shape=input_shape)\n")
    f.write("  Layer_0 = Conv2D(filters=filters, kernel_size=(3,3), strides=(1, 1),padding=\"same\",kernel_initializer=\"he_normal\")(input)\n")
    f.write("  MergedLayer_0 = BatchNormalization(axis=CHANNEL_AXIS)(Layer_0)\n")

def CreateModelToFile(OutputDir,Arc,num_layers,Input,Output,out_filters):
  Arc = np.array([int(x) for x in Arc.split(" ") if x])
  start_idx = 0
  pool_distance = num_layers // 3
  pool_index = [ pool_distance - 1 , 2 * pool_distance - 1]
  pool_param = 0

  for i in range(0,num_layers):
    ArcPart= Arc[start_idx:start_idx+i+1]
    MergeLayerstringBuilder(OutputDir, i, ArcPart, pool_param, pool_distance)
    if i in pool_index:
      PooledLayerStringBuilder(OutputDir, i, pool_param, pool_distance)
      pool_param += 1
      with open( OutputDir + '/Model.py', 'a' ) as f:
        f.write("  filters *= 2\n")
    start_idx = start_idx + i + 1
  with open( OutputDir + '/Model.py', 'a' ) as f:
    f.write("\n  FinalPooling = GlobalAveragePooling2D()(MergedLayer_" + str(num_layers) + ")\n")
    f.write("  dropout = Dropout(0.5)(FinalPooling)\n")
    f.write("  dense = Dense(units=num_outputs, kernel_initializer=\"he_normal\", activation=\"softmax\")(dropout)\n")
    f.write("  model = Model(inputs=input, outputs=dense)\n" )
    f.write("  return model\n")
  
