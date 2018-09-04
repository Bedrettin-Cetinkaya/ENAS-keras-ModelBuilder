import argparse
import numpy as np
import os
import createModel
import shutil
import importlib

def main():
  parser = argparse.ArgumentParser(description='Parse Arguments')
  parser.add_argument('--out_filters', type=int, default=96)
  parser.add_argument('--fixed_arc', type=str, default="None")
  parser.add_argument('--num_layers', type=int, default= 24)
  parser.add_argument('--output_dir', type=str, default="output")
  parser.add_argument('--reset_output_dir', type=int, default=1)
  args=parser.parse_args()

  if args.fixed_arc == "None":
    raise Exception('fixed_arc should be given!')

  if not os.path.isdir(args.output_dir):
    print("Path {} does not exist. Creating.".format(args.output_dir))
    os.makedirs(args.output_dir)
    shutil.copyfile('__init__.py',args.output_dir + "/__init__.py")
  elif args.reset_output_dir:
    print("Path {} exists. Remove and remake.".format(args.output_dir))
    shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir)
    shutil.copyfile('__init__.py', args.output_dir + "/__init__.py")

  #input size with batches, NHWC format
  img_rows, img_col, img_channel = 32, 32, 3
  #output classes number
  output = 10

  createModel.DefaultCodesToFile(args.output_dir)
  createModel.CreateModelToFile(args.output_dir,args.fixed_arc,args.num_layers,input,output,args.out_filters)
  ModelBuilder = importlib.import_module(args.output_dir + ".Model")

  Model = ModelBuilder.BuildModel([img_rows, img_col, img_channel], output, args.out_filters)
  print Model.summary()
if __name__ == '__main__':
  main()

