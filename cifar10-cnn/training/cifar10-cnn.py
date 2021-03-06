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
import os
import sys
import argparse

from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorpack.tfutils.varreplace import remap_variables
from tensorpack.dataflow import dataset
from tensorpack.utils.gpu import get_nr_gpu
import tensorpack.tfutils.symbolic_functions as symbf

from dorefa import get_dorefa

from evaluate import *
from loader import loader

MONITOR = 0

IMAGE_SIZE = 32

BITW = 1
BITA = 5
BITG = 32

class Model(ModelDesc):
    def __init__(self, cifar_classnum):
        super(Model, self).__init__()
        self.cifar_classnum = cifar_classnum

    def _get_inputs(self):
        return [InputDesc(tf.float32, (None, IMAGE_SIZE, IMAGE_SIZE, 3), 'input'),
                InputDesc(tf.int32, (None,), 'label')]

    def _build_graph(self, inputs):
        image, label = inputs


        fw, fa, fg = get_dorefa(BITW, BITA, BITG)

        old_get_variable = tf.get_variable

        def monitor(x, name):
            if MONITOR == 1:
                return tf.Print(x, [x], message='\n\n' + name + ': ', summarize=1000, name=name)
            else:
                return x

        def new_get_variable(v):
            name = v.op.name
            if not name.endswith('W') or 'conv0' in name or 'fc1' in name:
                return v
            else:
                logger.info("Quantizing weight {}".format(v.op.name))
                if MONITOR == 1:
                    return tf.Print(fw(v), [fw(v)], message='\n\n' + v.name + ', Quantized weights are:', summarize=100)
                else:
                    return fw(v)

        def bn_activate(name, x):
            X = BatchNorm(name, x)
            x = monitor(x, name + '_noact_out')
            return activate(x)

        def activate(x):
            if BITA == 32:
                return tf.nn.relu(x)
            else:
                return fa(tf.nn.relu(x))

        with remap_variables(new_get_variable), \
             argscope(Conv2D, kernel_shape=3, use_bias=False, nl=tf.identity, out_channel=32):
            logits = (LinearWrap(image).apply(monitor, 'image_out')
                      .Conv2D('conv0')
                      .apply(fg)
                      .BatchNorm('bn0')
                      .apply(activate).apply(monitor, 'conv0_out')
                      .MaxPooling('pool0', 2).apply(monitor, 'pool0_out')
                      .Conv2D('conv1')
                      .apply(fg)
                      .BatchNorm('bn1').apply(activate).apply(monitor, 'conv1_out')
                      .Conv2D('conv2')
                      .apply(fg)
                      .BatchNorm('bn2').apply(activate).apply(monitor, 'conv2_out')
                      .MaxPooling('pool1', 2).apply(monitor, 'pool1_out')
                      .FullyConnected('fc0', use_bias=False, out_dim=512, nl=tf.identity).apply(activate).apply(monitor, 'fc0_out')
                      .FullyConnected('fc1', use_bias=False, out_dim=self.cifar_classnum, nl=tf.identity).apply(monitor, 'fc1_out')
                      ())

        prob = tf.nn.softmax(logits, name='prob')

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')

        wrong = symbf.prediction_incorrect(logits, label, name='incorrect')
        accuracy = symbf.accuracy(logits, label, name='accuracy')

        train_error = tf.reduce_mean(wrong, name='train_error')
        summary.add_moving_summary(train_error, accuracy)

        wd_cost = tf.multiply(1e-5, regularize_cost('fc.*/W', tf.nn.l2_loss), name='regularize_loss')
        self.cost = tf.add_n([wd_cost, cost], name='total_cost')
        summary.add_moving_summary(cost, wd_cost, self.cost)

    def _get_optimizer(self):
        lr = get_scalar_var('learning_rate', 0.01, trainable=False, summary=True)
        opt = tf.train.AdamOptimizer(lr, epsilon=1e-3)
        return opt

def get_data(train_or_test, classnum):
    isTrain = train_or_test == 'train'
    if classnum == 10:
        ds = dataset.Cifar10(train_or_test)
    else:
        ds = dataset.Cifar100(train_or_test)
    #  if isTrain:
    #      augmentors = [
    #          imgaug.Flip(horiz=True),
    #          imgaug.Brightness(63),
    #          imgaug.Contrast((0.2, 1.8)),
    #          imgaug.MeanVarianceNormalize(all_channel=True)
    #      ]
    #  else:
    #      augmentors = [
    #          imgaug.MeanVarianceNormalize(all_channel=True)
    #      ]
    #  ds = AugmentImageComponent(ds, augmentors)
    ds = BatchData(ds, 256, remainder=not isTrain)
    return ds


def get_config(args):
    dataset_train = get_data('train', classnum=args.classnum)
    dataset_test = get_data('test', classnum=args.classnum)
    steps_per_epoch = dataset_train.size()

    return TrainConfig(
        model=Model(args.classnum),
        dataflow=dataset_train,
        callbacks=[
            ModelSaver(),
            MaxSaver('validation_accuracy'),
            InferenceRunner(
                dataset_test,
                [ScalarStats('cross_entropy_loss'), ScalarStats('accuracy'), ClassificationError('incorrect')]
            ),
        ],
        steps_per_epoch=steps_per_epoch,
        max_epoch=100,
    )

def run_test(weights_file, test_file, classnum):
    if MONITOR == 1:
        monitor_names = ['image_out', 'conv0_out', 'pool0_out', 'conv1_out', 'conv2_out', 'pool1_out', 'fc0_out', 'fc1_out']
    else:
        monitor_names = []
    output_names = ['prob']
    output_names.extend(monitor_names)

    param_dict = np.load(weights_file, encoding='latin1', allow_pickle=True).item()
    predictor = OfflinePredictor(PredictConfig(
        model=Model(classnum),
        session_init=DictRestore(param_dict),
        input_names=['input'],
        output_names=output_names
    ))

    if MONITOR == 1:
        NUM = 1
    else:
        NUM = 10000

    l = loader()
    l.load_cifar(test_file, num_samples=NUM, classes=classnum)
    images = np.zeros((NUM, IMAGE_SIZE, IMAGE_SIZE, 3))
    for i, ret in enumerate(l.ret):
        images[i,:,:,:] = ret[0]

    outputs = predictor([images])

    prob = outputs[0]

    correct_count = 0
    maxes = np.argmax(prob, axis=1)
    ## label and prediction
    for i, ret in enumerate(l.ret):  
        print("Golden: {}, predicted: {}".format(ret[1], maxes[i]))
        if ret[1] == maxes[i]:
            correct_count += 1

    print(str(correct_count) + ' correct out of ' + str(NUM))

    index = 1
    for o in monitor_names:
        print(o + ', shape: ' + str(outputs[index].shape) )

        if 'image' not in o and BITA >= 4:
            print(str(outputs[index]))

        if len(outputs[index].shape) == 4:
            file_name = o.split('/')[-1]
            print('Writing file... ' + file_name)
            if not os.path.exists('./log'):
                os.makedirs('./log')
            with open('./log/' + file_name + '.log', 'w') as f:
                for sample in range(0, outputs[index].shape[0]):
                    for h in range(0, outputs[index].shape[1]):
                        for w in range(0, outputs[index].shape[2]):
                            res = ''
                            for c in range(0, outputs[index].shape[3]):
                                if 'image' in file_name:
                                    res = hexFromInt( int(outputs[index][sample, h, w, c]), 8 ) + '_' + res
                                else:
                                    res = hexFromInt( int(outputs[index][sample, h, w, c]), BITA) + '_' + res
                            f.write('0x' + res + '\n')
        index += 1

def dump_weights(meta, model, output):
    fw, fa, fg = get_dorefa(BITW, BITA, BITG)

    with tf.Graph().as_default() as G:
        tf.train.import_meta_graph(meta)

        init = get_model_loader(model)
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        sess.run(tf.global_variables_initializer())
        init.init(sess)

        with sess.as_default():
            if output:
                if output.endswith('npy') or output.endswith('npz'):
                    varmanip.dump_session_params(output)
                else:
                    var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                    var.extend(tf.get_collection(tf.GraphKeys.MODEL_VARIABLES))
                    var_dict = {}
                    for v in var:
                        name = varmanip.get_savename_from_varname(v.name)
                        var_dict[name] = v
                    logger.info("Variables to dump:")
                    logger.info(", ".join(var_dict.keys()))
                    saver = tf.train.Saver(
                        var_list=var_dict,
                        write_version=tf.train.SaverDef.V2)
                    saver.save(sess, output, write_meta_graph=False)

            network_model = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            network_model.extend(tf.get_collection(tf.GraphKeys.MODEL_VARIABLES))

            target_frequency = 200000000
            target_FMpS = 300000
            non_quantized_layers = ['conv0/Conv2D', 'fc1/MatMul']

            json_out, layers_list, max_cycles = generateLayers(sess, BITA, BITW, non_quantized_layers, target_frequency, target_FMpS)
            
            achieved_FMpS = target_frequency/max_cycles

            generateConfig(layers_list, 'cifar10-cnn-config-{}W{}A.h'.format(BITW, BITA))
            genereateHLSparams(layers_list, network_model, 'cifar10-cnn-params-{}W{}A.h'.format(BITW, BITA), fw)

            print('|---------------------------------------------------------|')
            print('target_FMpS: ' + str(target_FMpS) )
            print('achieved_FMpS: ' + str(achieved_FMpS) )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dump2_train1_test0', help='dump(2), train(1) or test(0)')
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--model', help='model file')
    parser.add_argument('--meta', help='metagraph file')
    parser.add_argument('--weights', help='weights file')
    parser.add_argument('--testfile', help='mnist testfile')
    parser.add_argument('--output', help='output for dumping')
    parser.add_argument('--classnum', help='10 for cifar10 or 100 for cifar100',
                        type=int, default = 10)
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    print('dump2_train1_test0: ' + str(args.dump2_train1_test0) )

    if args.dump2_train1_test0 == '1':
        MONITOR = 0
        logger.auto_set_dir()
        config = get_config(args)
        if args.model:
            config.session_init = SaverRestore(args.model)
        SimpleTrainer(config).train()

    elif args.dump2_train1_test0 == '0':
        if args.weights == None:
            print('Provide weights file (.npy) for testing!')
            sys.exit()
        run_test(args.weights, args.testfile, args.classnum)

    elif args.dump2_train1_test0 == '2':
        if args.meta == None:
            print('Provide meta file (.meta) for dumping')
            sys.exit()
        if args.model == None:
            print('Provide model file (.data-00000-of-00001) for dumping')
            sys.exit()
        dump_weights(args.meta, args.model, args.output)
