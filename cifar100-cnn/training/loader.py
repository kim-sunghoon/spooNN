#*************************************************************************
# Copyright (C) 2018 Kaan Kara - Systems Group, ETH Zurich

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#*************************************************************************

import numpy as np
import six
import pickle
import argparse

class loader:
    def __init__(self):
        self.num_samples = 0;
        self.ret = []

    def load_cifar(self, path_to_file, num_samples, classes = 10):
        assert classes == 10 or classes == 100
        fo = open(path_to_file, 'rb')

        if six.PY3:
            dic = pickle.load(fo, encoding = 'bytes')
        else:
            dic = pickle.load(fo)

        data = dic[b'data']

        #  img_filename = dic[b'filenames']

        if classes == 10:
            label = dic[b'labels']
        else: 
            label = dic[b'fine_labels']
        fo.close()

        for k in range(num_samples):
            img = data[k].reshape(3, 32, 32)
            img = np.transpose(img, [1, 2, 0])
            #  self.ret.append([img, label[k], img_filename])
            self.ret.append([img, label[k]])
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--testfile', help='cifar testfile',
            default = 'cifar-100-python/test'
            )
    parser.add_argument('--num_samples', help='num samples to visualize, maximum is 10000',
            default = 1000
            )
    parser.add_argument('--num_classes', help='num classes, cifar10 for 10 or cifar100 for 100',
            default = 100
            )
    args = parser.parse_args()
    import cv2

    l = loader()
    l.load_cifar(path_to_file = args.testfile, num_samples = args.num_samples, classes = args.num_classes)
    
    for i, ret in enumerate(l.ret):
        img = ret[0]
        label = ret[1]
        #  img_filename = ret[2]
        cv2.imwrite("fig/{:04d}.jpg".format(i), img)
        #  print(label, img_filename)
        print(label)
