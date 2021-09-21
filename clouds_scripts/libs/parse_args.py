import argparse, warnings, datetime, os
import numpy as np
from .ServiceDefs import DoesPathExistAndIsFile


def parse_args(args):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--run-prefix', dest='run_prefix', default='devel')
    parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)', required=True)
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)', required=True)
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')

    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=240)

    parser.add_argument('--clf-weight', dest='clf_weight', help='The weight for the classification loss component', type=float, default=1.0)
    parser.add_argument('--rgr-weight', dest='rgr_weight', help='The weight for the regression loss component', type=float, default=1.0)

    parser.add_argument('--fl-alpha', dest='fl_alpha', help='Alpha coefficient in Focal loss', type=float, default=0.25)
    parser.add_argument('--fl-gamma', dest='fl_gamma', help='Gamma coefficient in Focal loss', type=float, default=2.0)

    parser.add_argument('--snapshot', help='snapshot of the network weights to start optimization from', type=str)

    return preprocess_args(parser.parse_args(args))


def preprocess_args(parsed_args):
    return parsed_args