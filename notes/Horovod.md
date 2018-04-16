# Horovod

* Horovod is a standalone Python package for deep learning training communication.
* It is a ring-allreduce implementation with Nvidia NCCL (MPI inter-node, I guess).
* It handles the situation in a server with multi GPUs, as well as single GPU.
* It uses broadcast to initialize model on all workers.

![alt text](https://github.com/zhaozhang/TACC_Deep_Learning_Reading/blob/master/Weekly_Notes/Horovod-figures/horovod-dance.png "Figure 1")
https://www.youtube.com/watch?v=-KZMPgJ3IoM

There requires only FOUR code changes in existing TensorFlow programs to use Horovod.



## Distributed Training Review
 ![alt text](https://github.com/zhaozhang/TACC_Deep_Learning_Reading/blob/master/Weekly_Notes/Horovod-figures/dis-training.png "Figure 2")

## From Scatter-Reduce to AllReduce
* This is bandwidth optimal, so it assumes every data transfer is bandwidth dominated. 
* If the data is small enough, the data transfer is latency dominated, then you may use recursive doubling algorithm for all-reduce. 
* May refer to Robert van de Geijn's works.

![alt text](https://github.com/zhaozhang/TACC_Deep_Learning_Reading/blob/master/Weekly_Notes/Horovod-figures/scatter-reduce.png "Figure 3")

![alt text](https://github.com/zhaozhang/TACC_Deep_Learning_Reading/blob/master/Weekly_Notes/Horovod-figures/complexity.png "Figure 4")

## Code Example
### Outline
```python
import tensorflow as tf
import horovod.tensorflow as hvd

# Initialize Horovod
hvd.init()

# Pin GPU to be used to process local rank (one GPU per process)
config = tf.ConfigProto()
config.gpu_options.visible_device_list = str(hvd.local_rank())

# Build model…
loss = …
opt = tf.train.AdagradOptimizer(0.01)

# Add Horovod Distributed Optimizer
opt = hvd.DistributedOptimizer(opt)

# Add hook to broadcast variables from rank 0 to all other processes during
# initialization.
hooks = [hvd.BroadcastGlobalVariablesHook(0)]

# Make training operation
train_op = opt.minimize(loss)

# The MonitoredTrainingSession takes care of session initialization,
# restoring from a checkpoint, saving to a checkpoint, and closing when done
# or an error occurs.
with tf.train.MonitoredTrainingSession(checkpoint_dir=“/tmp/train_logs”,
                                      config=config,
                                      hooks=hooks) as mon_sess:
 while not mon_sess.should_stop():
   # Perform synchronous training.
   mon_sess.run(train_op)
```

### Keras Mnist Example
```python
from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import math
import tensorflow as tf
import horovod.keras as hvd

# Horovod: initialize Horovod.
hvd.init()

# Horovod: pin GPU to be used to process local rank (one GPU per process)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank())
K.set_session(tf.Session(config=config))

batch_size = 128
num_classes = 10

# Horovod: adjust number of epochs based on number of GPUs.
epochs = int(math.ceil(12.0 / hvd.size()))

# Input image dimensions
img_rows, img_cols = 28, 28

# The data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Horovod: adjust learning rate based on number of GPUs.
opt = keras.optimizers.Adadelta(1.0 * hvd.size())

# Horovod: add Horovod Distributed Optimizer.
opt = hvd.DistributedOptimizer(opt)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=opt,
              metrics=['accuracy'])

callbacks = [
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),
]

# Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
if hvd.rank() == 0:
    callbacks.append(keras.callbacks.ModelCheckpoint('./checkpoint-{epoch}.h5'))

model.fit(x_train, y_train,
          batch_size=batch_size,
          callbacks=callbacks,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
```


## Keras + TensorFlow + Horovod on Stampede2 KNL

### Keras Installation
```
pip install --user keras
```

### TensorFlow Installation
```
pip install --user /scratch/00946/zzhang/stampede2/tensorflow/build/tensorflow-1.5.0-cp27-cp27mu-linux_x86_64.whl
```

### Horovod Installation
```
pip install --user horovod
```

### TensorFlow Benchmarking
```
git clone https://github.com/tensorflow/benchmarks.git
cd benchmarks

mpiexec.hydra -l -ppn 1 -n 2 taskset -c 0-64,68-132,136-200,204-268 numactl -p 1 python scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py --model resnet50 --batch_size 128 --variable_update horovod --data_format NCHW --num_intra_threads 64 --num_inter_threads 3 --distortions=False --num_batches 200

# Should be able to see 70 imgs/sec on 1 KNL, 140 imgs/sec on 2KNLs
# A 100-category real data is in 
#/work/00946/zzhang/imagenet/imagenet-p100-benchmark/tensorflow/train-00000-of-00001 
#and /work/00946/zzhang/imagenet/imagenet-p100-benchmark/tensorflow/validation-00000-of-00001
# Should be able to see 62-64 imgs/sec on 1 KNL
```

### Keras Mnist Example
```
git clone https://github.com/uber/horovod.git
cd examples/
mpiexec.hydra -l -ppn 1 -n 1 taskset -c 0-64,68-132,136-200,204-268 numactl -p 1 python keras_mnist.py
#I see funny results, need to debug a bit.
```