import argparse
import sys

from experiments import *

parser = argparse.ArgumentParser(description='Run the GAN!')
parser.add_argument('-m', '--model', type=str, help='Choose a model to run')
parser.add_argument('-e', '--epoch', type=int, help='set epoch size', default=50)
m_args = parser.parse_args()

if m_args.model is None:
    sys.exit("You need to give a model name with -m")

if m_args.model == 'gan':
    gan.run_experiment(m_args.epoch)

elif m_args.model == 'dcgan':
    dcgan.run_experiment(m_args.epoch)



