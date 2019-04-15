from pgan import PGAN
import argparse
import os

parser = argparse.ArgumentParser(description='Train Progressive GAN')
parser.add_argument('-d', dest='d_path', help='path for the dataset')
parser.add_argument('-s', dest='s_path', help='snapshot path')
ARGS = parser.parse_args()
def main(data_path, save_path):
    gan = PGAN(data_path, save_path)
    gan.train(4, [128, 64, 32, 16])
if __name__ == "__main__":
    if not os.path.isdir(ARGS.s_path):os.makedirs(ARGS.s_path)
    main(ARGS.d_path, ARGS.s_path)