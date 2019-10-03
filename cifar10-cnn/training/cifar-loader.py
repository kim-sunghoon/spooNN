import numpy as np
import argparse
import pickle

def cifar_loader(path_to_file):
    with open(path_to_file, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
    return dict
    
if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--testfile', help='cifar testfile')
    args = parser.parse_args()

    cifar_dict = cifar_loader(args.testfile)

    #  print(cifar_dict)

    print(type(cifar_dict))
    pprint.pprint(cifar_dict)
