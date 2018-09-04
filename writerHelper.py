import numpy as np

def MergeLayerstringBuilder(OutputDir, LayerIndex, ArcPart, PoolParam, PoolDistance):

  #LayerKeywords = "Layer_"
  InputLayerKeywords = "MergedLayer_"
  if LayerIndex % PoolDistance == 0 :
    if PoolParam == 1:
      InputLayerKeywords = "PooledLayer_"
    elif PoolParam == 2:
      InputLayerKeywords = "S_PooledLayer_"


  MergedLayerKeywords = ["MergedLayer_", "PooledLayer_", "S_PooledLayer_", "T_PooledLayer_"]

  Char = "  #"
  with open(OutputDir + "/Model.py", "a") as f:
    f.write( "\n" + Char  + " ".join(str(x) for x in ArcPart) )
    if ArcPart[0] in [0, 1]:
      f.write("\n  Layer_" + str(LayerIndex + 1) + " = _conv_bn_relu(filters=filters, kernel_size=(3,3), strides=(1, 1), padding=\"same\")(" + InputLayerKeywords +
      str(LayerIndex) + ")")

    elif ArcPart[0] in [2, 3]:
      f.write("\n  Layer_" + str(LayerIndex + 1) + " = _conv_bn_relu(filters=filters, kernel_size=(5,5), strides=(1, 1), padding=\"same\")(" + InputLayerKeywords +
      str(LayerIndex) + ")")

    elif ArcPart[0] in [4]:
      f.write("\n  Layer_" + str(LayerIndex + 1) + " = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding=\"same\")(" + InputLayerKeywords +
      str(LayerIndex) + ")")

    elif ArcPart[0] in [5]:
      f.write("\n  Layer_" + str(LayerIndex + 1) + " = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding=\"same\")(" + IputLayerKeywords +
      str(LayerIndex) + ")")

    if LayerIndex == "1":
      f.write("\n  MergedLayer_1 = Layer_1")
    MergedString = ""

    for i in range(1,LayerIndex + 1):
      if ArcPart[i] == 1:
        if i <= (PoolParam * PoolDistance):
          MergedString += MergedLayerKeywords[ PoolParam ] + str(i) + ", "
        else :
          MergedString += MergedLayerKeywords[0] + str(i) + ", "
    MergedString += "Layer_" + str(LayerIndex + 1) + ", " + "filters"
    f.write("\n  MergedLayer_" + str(LayerIndex + 1) +  " =  MergeLayers(" + MergedString  + ")" + "\n")




def PooledLayerStringBuilder(OutputDir, LayerIndex, PoolParam, PoolDistance):

  BeforePooledLayerKeywords = ["MergedLayer_", "PooledLayer_" , "S_PooledLayer_" , "T_PooledLayer_"]
  AfterPooledLayerKeywords =  ["  PooledLayer_" , "  S_PooledLayer_" , "  T_PooledLayer_", "  F_PooledLayer_"]

  KeyIndex = 0
  PoolCounter = 1
  with open(OutputDir + "/Model.py", "a") as f:
    for i in range(LayerIndex+1,0,-1):
      f.write("\n" + AfterPooledLayerKeywords[PoolParam] + str(i) + " = PoolLayer(" + BeforePooledLayerKeywords[KeyIndex] + str(i) +  ", filters)" )
      if PoolCounter == PoolDistance:
        KeyIndex += 1
        PoolCounter = 0
      PoolCounter += 1
    f.write("\n")
