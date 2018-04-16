# Deep Learning at TACC

## Overview: 
Scientists from many domains are actively exploring and adopting deep learning as a cutting-edge methodology to make research breakthrough. 
At the Texas Advanced Computing Center (TACC), our mission is to enable discoveries that advance science and society through the application of advanced computing technologies.
Thus, we are embracing this new type of application on our high end computing platforms.
Technically, we have Caffe, TensorFlow, and MXNet ready on our Stampede2 supercomputer (Intel KNL) and the Maverick cluster (Nvidia K40).
We encourage users to try these tools and give us feedback on the usage experience. 

## Available Deep Learning Tools:
We have made three popular platforms available. 
Caffe is available on both Stampede2 and Maverick.
TensorFlow and MXnet are available only on Maverick.

### Caffe (Intel Distribution)
https://github.com/intel/caffe

### TensorFlow
https://www.tensorflow.org

### MXNet
http://mxnet.io

## Quick Start
Please follow the following instructions for your first test execution (Cifar10 test example).

### Caffe
#### Stampede2 Single Node
1. Get on a compute node
```
idev -A PROJECT -q QUEUE -N 1 -n 1 -t 01:00:00
```

2. Copy Model and Data
You may copy the test directory to other place.
```
cp -r /scratch/00946/zzhang/stampede2/test $SCRATCH/
```

3. Enter the Directory
```
cd $SCRATCH/test
```

4. Load Caffe Module
```
module load caffe
```

5. Train the Model
```
caffe.bin train -engine "MKL2017" --solver=examples/cifar10/cifar10_full_solver.prototxt
```

#### Stampede2 Multiple Nodes
1. Get on a compute node
```
idev -A PROJECT -q QUEUE -N 4 -n 4 -t 01:00:00
```

2. Copy Model and Data
You may copy the test directory to other place.
```
cp -r /scratch/00946/zzhang/stampede2/test $SCRATCH/
```

3. Enter the Directory
```
cd $SCRATCH/test
```

4. Load Caffe Module
```
module load caffe
```

5. Train the Model
```
ibrun -np 4 caffe.bin train -engine "MKL2017" --solver=examples/cifar10/cifar10_full_solver.prototxt
```

Notes: the -np option in the ibrun command has to be exactly the same as the -N and -n option in the idev or slurm script. The point is to launch one process on each node. The Intel Caffe uses OpenMP for single node parallelism and MPI for multiple-node parallelism. 

Launching the cifar10 example without chaning the batch size in examples/cifar10/cifar10_full_solver.prototxt
is weak scaling. That is to say, by increasing the number of nodes, we are also increasing the batch size.
If you are thinking about strong scaling, please decrease the batch size accordingly.

From our experience, weak scaling can achieve ~80% efficiency on 512 KNL nodes while the strong scaling efficiency drops at ~50% on 8 KNL nodes.


#### Maverick Single Node
1. Get on a compute node
```
idev -t 01:00:00
```

2. Copy Model and Data
You may copy the test directory to other place.
```
cp -r /work/00946/zzhang/maverick/test $WORK/
```

3. Enter the Directory
```
cd $WORK/test
```

4. Load Caffe Module
```
ml gcc/4.9.3  cuda/8.0  cudnn/5.1 python/2.7.12 boost/1.59 caffe
```

5. Train the Model
```
caffe.bin train --solver=caffe-examples/cifar10/cifar10_full_solver.prototxt
```

Note: Multiple-node training on Maverick is not ready yet, it is coming soon.

### TensorFlow (v1.0.0)
1. Get on a compute node
```
idev -t 01:00:00
```

2. Copy Model and Data
You may copy the test directory to other place.
```
cp -r /work/00946/zzhang/maverick/test $WORK/
```

3. Enter the Directory
```
cd $WORK/test
```

4. Load the module
```
module load tensorflow-gpu
```

5. Train the Model
```
python tf-examples/tf_cnn_benchmarks.py --model=alexnet
```

### MXNet (v0.10.0-79-g790328f)
1. Get on a compute node
```
idev -t 01:00:00
```

2. Copy Model and Data
You may copy the test directory to other place.
```
cp -r /work/00946/zzhang/maverick/test $WORK/
```

3. Enter the Directory
```
cd $WORK/test
```

4. Load the module
```
module load mxnet
```

5. Train the Model
```
python mxnet-examples/image-classification/train_cifar10.py --network resnet --gpus 0
```

## Python Interface
1. Module purge
```
module purge
```

2. Load Caffe Module
```
ml gcc/4.9.3  cuda/8.0  cudnn/5.1 python/2.7.12 boost/1.59 caffe
```

3. Start Python
If correct, it should be python/2.7.12 on Maverick
```
python
>>> import caffe
>>> print caffe.__file__
```
Now if you would like to use TensorFlow and MXNet on Maverick, repeat 1 and 2, then load the tensorflow and mxnet module, respectively.

3. Load Tensorflow Module and Run Python
```
module load tensorflow-gpu
python
>>> import tensorflow as tf
>>> print tf.__file__
```

3. Load MXnet Module and Run Python
```
module load mxnet
python
>>> import mxnet
>>> print mxnet.__file__
```

Notes: If you have missing packages in Python, install it to user space by running:
```
pip install --user package_name
```


## Performance Evaluation
xxx