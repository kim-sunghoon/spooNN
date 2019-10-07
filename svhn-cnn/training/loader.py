
import numpy as np
import argparse
import scipy.io
import cv2

class loader:
    def __init__(self):
        self.ret = []
        self.img = []
        self.label = []

    def load_svhn(self, path_to_file, num_samples):
        data = scipy.io.loadmat(path_to_file)
        self.img = data['X'].transpose(3,0,1,2)
        self.label = data['y'].reshape((-1))
        self.label[self.label == 10] = 0

        total_num_data = self.img.shape[0]

        for i in range(0, num_samples):
            self.ret.append([self.img[i], self.label[i]])
        print("{} out of {} data processed".format(num_samples, total_num_data))
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--testfile', help='testfile',
            default = 'svhn_data/test_32x32.mat'
            )
    parser.add_argument('--num_samples', help='num samples to visualize, maximum is 10000',
            default = 1
            )
    args = parser.parse_args()

    l = loader()
    l.load_svhn(args.testfile, args.num_samples)
    
    for i, ret in enumerate(l.ret):
        img = ret[0]
        label = ret[1]
        cv2.imwrite("fig/{:04d}.jpg".format(i), img)
        #  print(label, img_filename)
        print(label)
