# Semantic Segmentation
### Introduction
In this project, I used techniques like Semantic segmentation and inference optimization to label the pixels of a road in images using a Fully Convolutional Network (FCN). Semantic segmentation identifiesfree space on the road at pixel-level granularity. 

### Setup
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

##### Run
Run the following command to run the project:
```
python main.py
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

### Submission
1. Ensure you've passed all the unit tests.
2. Ensure you pass all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view).
3. Submit the following in a zip file.
 - `helper.py`
 - `main.py`
 - `project_tests.py`
 - Newest inference images from `runs` folder

### Implementation
The FCN structurally comprises of a encoder and a decoder. The encoder is a series of convolutional layers like VGG and its goal is to extract features from the image. The decoder up-scales the output of the decoder such that it is the same size as the original image. Thus, resulting in segmentation or prediction of each individual pixel in the original image. 
The implementation starts with a pre trained canonical model VGG and removes its final, fully-connected layers, to add techniques like 1x1 convolutions, upsampling, and skip layers to train the FCN.The result is an FCN which classifies each road pixel in the image.   
##### Load the pre trained vgg model
The `load_vgg` function in `main.py` helps load a pre trained vgg model('https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip') into TensorFlow and returns a tuple of Tensors from the vgg model. 
```
def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    
    tf.saved_model.loader.load(sess, ['vgg16'], vgg_path)
    graph = tf.get_default_graph()
    return graph.get_tensor_by_name(vgg_input_tensor_name), \
            graph.get_tensor_by_name(vgg_keep_prob_tensor_name), \
            graph.get_tensor_by_name(vgg_layer3_out_tensor_name), \
            graph.get_tensor_by_name(vgg_layer4_out_tensor_name), \
            graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
tests.test_load_vgg(load_vgg, tf)

```

#### Learn the correct features from the images
The `layers` function in `main.py` helps 
1. replace fully connected layers with 1x1 convolutional layers 
2. up-sampling through the use of transposed convolutional layers and 
3. skip connection architecture that allows images to use information from multiple resolution scales as a result the network is able to make more precise segmentation decisions.

The 1x1 convolutional layer results in output value with the tensor will remain 4D instead of flattening to 2D, thereby preserving spatial information. Here is an example from the code depicting converting fully connected layers to 1x1 convolutional layers.
```

    # 1x1 conv layer 7, 5x18xnum_classes
    layer_7_conv = tf.layers.conv2d_transpose(vgg_layer7_out, 64, (1,1), (1,1), name='1x1_vgg_7',kernel_initializer=tf.truncated_normal_initializer(stddev=sigma))
    # 1x1 conv layer 4, 10x36xnum_classes
    layer_4_conv = tf.layers.conv2d_transpose(vgg_layer4_out, 64, (1,1), (1,1), name='1x1_vgg_4',kernel_initializer=tf.truncated_normal_initializer(stddev=sigma))
    # 1x1 conv layer 3, 20x72xnum_classes
    layer_3_conv = tf.layers.conv2d_transpose(vgg_layer3_out, 64, (1,1), (1,1), name='1x1_vgg_3',kernel_initializer=tf.truncated_normal_initializer(stddev=sigma))

```

The Transposed convolution is a reverse convolution in which the forward and backward passes are swapped.Transposed convolutions are used to upsample the input and are a core part of the FCN architecture.In TensorFlow, the API tf.layers.conv2d_transpose is used to create a transposed convolutional layer. Here is an example of the code depicting use of transpose convolution to upsample

```
# Decoder layer 1, 10x36xnum_classes
    decoder_layer_1 = tf.layers.conv2d_transpose(layer_7_conv, 64, upsample_kernel_size, upsample_stride_size, name='decoder_1',kernel_initializer=tf.truncated_normal_initializer(stddev=sigma))

``` 

One effect of convolutions in general is that it narrows the scope by looking closely at some feature and loses the bigger picture as a result. As a result, the decoder of the encoder input back to the original size may cause some information to be lost. Skip connections are a way of retaining that information easily.The skip connections works by connecting the output of one layer to a non adjacent layer. Too many skip connections however can explode the size of the model. For VGG-16 typically only the third and fourth pooling layers are used for skip connections. Here is an example of the code depicting use of skip connections

```
    skip1 = tf.add(decoder_layer_1, layer_4_conv)

```

#### Optimize the neural network

The `optimize` function in `main.py` depicts how the TensorFLow loss and optimizer operations are built to help optimize the network. An Adam optimizer with an initial learning rate of .0001 is used for training. The goal is to reduce the loss of information.

```
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # The output tensor is 4D so we have to reshape it to 2D. logits is now a 2D tensor where each row represents a pixel and each column a class.
    logits = tf.reshape(nn_last_layer, (-1, num_classes))

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label)
    cross_entropy_loss = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss)

```

#### Train the Neural network
One of the challenges of semantic segmentation is that it requires a lot of computational power. Inference optimization techniques such as fusion, quantization and reduced precision help accelerate network performance.
Here is the implementation of the train_nn function that get batches of training data and uses the batch size and epoch hyper parameters to get training data to train the neural network and print out the loss during the training

```
def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, augment, images):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    global LEARNING_RATE
    global KEEP_PROB
    sample=0
    loss_history = []
    for epoch in range(epochs):
        for image, image_c in get_batches_fn(batch_size):
            augmented = sess.run([augment], feed_dict={
                    images: image
            })
            _,loss = sess.run([train_op, cross_entropy_loss], feed_dict={
                input_image: augmented[0],
                correct_label: image_c,
                keep_prob: KEEP_PROB,
                learning_rate: LEARNING_RATE
            })
            sample = sample + batch_size
        print('Epoch {} of {} - Loss: {}'.format(epoch, epochs, loss))
        loss_history.append(loss)
    return loss_history
tests.test_train_nn(train_nn)

```
#### Hyper parameter settings
```
#How many times to run the model on our training data through the network. The higher this value the better the accuracy but it impacts the training speed
EPOCHS = 15

#How many images to run at a time in tensorflow during training. The larger the batch size the faster the model can train but the processory may have a memory limit larger the batch size
BATCH_SIZE = 8
```
### Conclusion

#### Training results
On average, the model decreases loss over time

[![SDC - PID controller d=0.1 ](https://github.com/bhatiarajesh/CarND-Semantic-Segmentation/raw/master/out/conclusion.png)]

### Newest inference images from the `runs` folder

Here are some example of free road space identified based on the training data and applying semantic segmantation techniques based on one run of my model

Example output #1
[![SDC - Semantic-Segmentation d=0.1 ](https://github.com/bhatiarajesh/CarND-Semantic-Segmentation/raw/master/out/um_000015.png)]

Example output #2
[![SDC - Semantic-Segmentation d=0.1 ](https://github.com/bhatiarajesh/CarND-Semantic-Segmentation/raw/master/out/um_000012.png)]

 
