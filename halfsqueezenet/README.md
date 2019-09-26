# halfsqueezenet: Object Detection for the System Design Contest at DAC 2018
This is the project that has delivered the highest frames per second processing rate (~25 FPS) at lowest power consumption with 50% intersection over union (IOU) bounding box accuracy. Using the contents of this project, an end-to-end flow from training to HLS implementation to hardware deployment can be reproduced.

![picture](drone.png)
![picture](car.png)
![picture](boat.png)
![picture](paraglider.png)
## Repo Organization
- training: Contains the training script and pre-trained weights and pre-generated header files to be used in Vivado HLS.
- hls: Contains the Vivado HLS implementation of halfsqueezenet, using the layers from hls-nn-lib.
- scripts: The scripts to generate RTL using Vivado HLS and to create a Vivado project to obtain the final bitstream.
- deploy: A Jupyter notebook showing how to use the FPGA based neural network to perform object detection on PYNQ.

## 1. Training

For training the CNN, you need to first install Tensorflow (https://www.tensorflow.org/install/).

If you haven't done so yet, clone the repo and go to the training directory:

1. $ git clone https://github.com/fpgasystems/spooNN
2. $ export PYTHONPATH=/path/to/spooNN/hls-nn-lib/training:$PYTHONPATH
3. $ cd spooNN/halfsqueezenet/training/

Have a look at the halfsqueezenet_objdetect.py script. The CNN that we will train will be quantized (1-bit weights and 5-bit activations).
```python
690         print('Start')                                                                                                       
691                                                                                                                              
692         parser = argparse.ArgumentParser()                                                                                   
693         parser.add_argument('dump2_train1_test0', help='dump(2), train(1) or test(0)')                                       
694         parser.add_argument('--model', help='model file')                                                                    
695         parser.add_argument('--meta', help='metagraph file')                                                                 
696         parser.add_argument('--output', help='output for dumping')                                                           
697         parser.add_argument('--gpu', help='the physical ids of GPUs to use')                                                 
698         parser.add_argument('--data', help='DAC dataset dir')                                                                
699         parser.add_argument('--run', help='directory of images to test')                                                     
700         parser.add_argument('--weights', help='weights file')                                                                
701         args = parser.parse_args()                                                                                           
702                                                                                                                              
703         print('Using GPU ' + str(args.gpu))                                                                                  
704                                                                                                                              
705         if args.gpu:                                                                                                         
706                 os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu                                                                
707                                                                                                                              
708         print(str(args.dump2_train1_test0))                                                                                  
709                                                                                                                              
710         if args.dump2_train1_test0 == '1':                                                                                   
711                 if args.data == None:                                                                                        
712                         print('Provide DAC dataset path with --data')                                                        
713                         sys.exit()                                                                                           
714                                                                                                                              
715                 config = get_config()                                                                                        
716                 if args.model:
717                         config.session_init = SaverRestore(args.model)                                                                                                                                                                                                                                                                                                                   
718                                                                                                                                                                                                                                                                                                                                                                                          
719                 SimpleTrainer(config).train()                                                                                                                                                                                                                                                                                                                                            
720                                                                                                                                                                                                                                                                                                                                                                                          
721         elif args.dump2_train1_test0 == '0':                                                                                                                                                                                                                                                                                                                                             
722                 if args.run == None:                                                                                                                                                                                                                                                                                                                                                     
723                         print('Provide images with --run ')                                                                                                                                                                                                                                                                                                                              
724                         sys.exit()                                                                                                                                                                                                                                                                                                                                                       
725                 if args.weights == None:                                                                                                                                                                                                                                                                                                                                                 
726                         print('Provide weights file (.npy) for testing!')                                                                                                                                                                                                                                                                                                                
727                         sys.exit()                                                                                                                                                                                                                                                                                                                                                       
728                                                                                                                                                                                                                                                                                                                                                                                          
729                 assert args.weights.endswith('.npy')                                                                                                                                                                                                                                                                                                                                     
730                 run_image(Model(), DictRestore(np.load(args.weights, encoding='latin1', allow_pickle=True).item()), args.run)                                                                                                                                                                                                                                                            
731                                                                                                                                                                                                                                                                                                                                                                                          
732         elif args.dump2_train1_test0 == '2':                                                                                                                                                                                                                                                                                                                                             
733                 if args.meta == None:                                                                                                                                                                                                                                                                                                                                                    
734                         print('Provide meta file (.meta) for dumping')                                                                                                                                                                                                                                                                                                                   
735                         sys.exit()                                                                                                                                                                                                                                                                                                                                                       
736                 if args.model == None:                                                                                                                                                                                                                                                                                                                                                   
737                         print('Provide model file (.data-00000-of-00001) for dumping')                                                                                                                                                                                                                                                                                                   
738                         sys.exit()                                                                                                                                                                                                                                                                                                                                                       
739                                                                                                                                                                                                                                                                                                                                                                                          
740                 dump_weights(args.meta, args.model, args.output)                                                                                                                                                                                                                                                                                                                         
741                                                                                                                                                                                                                                                                                                                                                                                          
742         elif args.dump2_train1_test0 == '3':                                                                                                                                                                                                                                                                                                                                             
743                 if args.run == None:                                                                                                                                                                                                                                                                                                                                                     
744                         print('Provide image with --run ')                                                                                                                                                                                                                                                                                                                               
745                         sys.exit()                                                                                                                                                                                                                                                                                                                                                       
746                 if args.weights == None:                                                                                                                                                                                                                                                                                                                                                 
747                         print('Provide weights file (.npy) for testing!')                                                                                                                                                                                                                                                                                                                
748                         sys.exit()                                                                                                                                                                                                                                                                                                                                                       
749                                                                                                                                                                                                                                                                                                                                                                                          
750                 assert args.weights.endswith('.npy')                                                                                                                                                                                                                                                                                                                                     
751                 run_single_image(Model(), DictRestore(np.load(args.weights, encoding='latin1', allow_pickle=True).item()), args.run)                                                                                                                                                                                                                                                     
```
Start the training (DAC18 object detection dataset is not publicly available yet):
- $ python ./halfsqueezenet_objdetect.py 1 --data /path/to/DAC18_object_detection_dataset

Generate the weights to be used in C and dump them also as a numpy array to weights.npy (replace NAME1 and NAME2 accordingly. NAME1 will be unique, whereas for NAME2 you can select weights from a certain iteration):
- $ python ./halfsqueezenet_objdetect.py 2 --meta ./train_log/halfsqueezenet_objdetect/NAME1.meta --model ./train_log/halfsqueezenet_objdetect/NAME2.data-00000-of-00001 --output weights.npy

This operation creates 2 files: halfsqueezenet-config.h and halfsqueezenet-params.h. The -config.h file contains layer-wise configuration parameters, for example the size of the convolution kernel, stride etc. The -params.h file contains the weights and activation factors as C arrays.

Perform inference using the weights you dumped. This will display the object detection result on all the images in the given directory.
- $ python ./halfsqueezenet_objdetect.py 0 --run path/to/DAC18_object_detection_dataset/category/ --weights ./weights.npy

## 2. HLS implementation and testing

Now that we have trained a network and have the weights as C arrays, we can implement the entire CNN using the layers provided in hls-nn-lib. We can also test the functionality of our CNN entirely, by compiling the design with gcc. To be able to compile this file, you need to have Vivado HLS installed.

![picture](folding_structure.png)

1. Take a look at `halfsqueezenet_folded.cpp`. This implementation is more complex than the mnist-cnn example, although we are still using the layers from hls-nn-lib. The reason for the complexity is that the entire CNN for object detection does not fit onto the FPGA fabric, so we have to fold the compute, as shown in the figure above. We use demuxes and muxes to guide the dataflow, depending on which part of the CNN we want to execute with an initiation.
2. Export XILINX_VIVADO to point to your installation: $ `export XILINX_VIVADO=/path/to/Xilinx/Vivado/2018.2`
3. Compile halfsqueezenet_folded.cpp with: /path/to/spooNN/halfsqueezenet/hls$ make
4. Test the executable with: /path/to/spooNN/halfsqueezenet/hls$ ./t
5. Running this will also generate a file "weights_file.txt". This is a file we need when running the CNN on the FPGA, since parts of weights will be transferred onto the FPGA during runtime.

## 3. RTL and bitstream generation

We now have a functionally correct CNN implemented in C, targeting Vivado HLS. 

1. Now we need to use Vivado HLS to generate RTL from the C implementation. There is a script for doing this: /path/to/spooNN/halfsqueezenet$ `./scripts/make_IP.sh /path/to/spooNN/halfsqueezenet`
2. After the RTL is generated it is packaged in an IP, that is located in /path/to/spooNN/halfsqueezenet/output/hls_project/sol1/impl/ip/xilinx_com_hls_halfsqueezenet_1_0.zip
3. Create a directory called repo: $ `mkdir repo`
4. Copy paste the IP into the new directory: `/path/to/spooNN/halfsqueezenet$ cp /path/to/spooNN/halfsqueezenet/output/hls_project/sol1/impl/ip/xilinx_com_hls_halfsqueezenet_1_0.zip ./repo`
5. Extract the IP: /path/to/spooNN/halfsqueezenet$ `unzip ./repo/xilinx_com_hls_halfsqueezenet_1_0.zip`

Now, the repo directory can be included in Vivado and the IP we generated can be instantiated.

1. We have a script to generate a final bitstream using this IP: `/path/to/spooNN/halfsqueezenet$ ./scripts/make_bitstream.sh /path/to/spooNN/halfsqueezenet`
2. This will take a while (around 30 minutes). After it is finished, find the generated bitstream in `/path/to/spooNN/halfsqueezenet/output/pynq-vivado/pynq-vivado.runs/impl_1/procsys_wrapper.bit`

Besides the bitstream, you also need a .hwh file that describes the overlay. 
3. You can find hw_handoff file at `$Path_To_Your_RTL_Project/$Project_Name/$Project_Name.srcs/sources_1/bd/design_1/hw_handoff`

i.e. `/path/to/spooNN/halfsqueezenet/output/pynq-vivado.srcs/sources_1/bd/procsys/hw_handoff/procsys.hwh`


## 4. Deployment on the PYNQ

Now that we have a `.bit` and a `.hwh` file (together, called an overlay), we show how to deploy it on the PYNQ.

1. Follow the guide to bring up you PYNQ (http://pynq.readthedocs.io) and make sure you can connect to it via SSH.
2. Copy the halfsqueezenet/deploy directory to the jupyter_notebooks directory on the PYNQ (assuming the static IP 192.168.2.99): $ scp -r /path/to/spooNN/halfsqueezenet/deploy xilinx@192.168.2.99:/home/xilinx/jupyter_notebooks/
3. Then open a browser window and navigate to http://192.168.2.99:9090, to access jupyter notebooks that are on the PYNQ.
4. Open deploy/halfsqueezenet.ipynb and follow the notebook to see how we can use the CNN on the FPGA to perform object detection. (overlay.bit and overlay.tcl are the files that are readily generated following the instructions of the previous part.)
