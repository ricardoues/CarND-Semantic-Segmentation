#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    img_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    l3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    l4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    l7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return img_input, keep, l3, l4, l7
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    
    vgg_layer7_out = tf.stop_gradient(vgg_layer7_out)
    vgg_layer4_out = tf.stop_gradient(vgg_layer4_out)
    vgg_layer3_out = tf.stop_gradient(vgg_layer3_out)    
    
    
    with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
        
        conv_1x1_1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding='same', 
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))  
        
        output_1 = tf.layers.conv2d_transpose(conv_1x1_1, num_classes, 4, 2, padding='same', 
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))       
    
        pool4_out_scaled = tf.multiply(vgg_layer4_out, 0.01)


        conv_1x1_2 = tf.layers.conv2d(pool4_out_scaled, num_classes, 1, padding='same', 
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))  
    
        comb_layer_1 = tf.add(output_1, conv_1x1_2)
    
        fcn_layer_1 = tf.layers.conv2d_transpose(comb_layer_1, num_classes, 4, 2, padding='same', 
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))  
    


        pool3_out_scaled = tf.multiply(vgg_layer3_out, 0.0001)

        
        conv_1x1_3 = tf.layers.conv2d(pool3_out_scaled, num_classes, 1, padding='same', 
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))  
    
        comb_layer_2 = tf.add(conv_1x1_3, fcn_layer_1)
        
        output_2 = tf.layers.conv2d_transpose(comb_layer_2, num_classes, 16, 8, padding='same', 
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))               
    
    return output_2
        
        
    
tests.test_layers(layers)




def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """


    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    
    correct_label = tf.reshape(correct_label, (-1,num_classes))
    
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label)
    
    cross_entropy_loss = tf.reduce_mean(cross_entropy)
    
    # I took ideas from the following web pages: 
    # https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer
    # https://stackoverflow.com/questions/33788989/tensorflow-using-adam-optimizer
    # https://discussions.udacity.com/t/how-to-ensure-correct-label-and-logits-are-of-the-same-size/618141
    
    decoder = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "decoder")
    
    
    
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    
    if len(decoder) == 0: 
        train_op = optimizer.minimize(cross_entropy_loss)
    else:
        
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_constant = 0.0001
        loss = cross_entropy_loss + reg_constant * sum(reg_losses)
        
        
        train_op = optimizer.minimize(loss, var_list=decoder)
                      
    
    return logits, train_op, cross_entropy_loss
            
    
tests.test_optimize(optimize)



def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
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
    :return 
    """
    sess.run(tf.global_variables_initializer())
    
    
    step = 0
    
    for epoch in range(epochs):
        # Loop over all batches 
        for images, label in get_batches_fn(batch_size):
            
            step += 1
            
            feed = { input_image: images, 
                     correct_label: label,
                     keep_prob: 0.5, 
                     learning_rate: 0.001}
            
            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict=feed)
            
            if step % 50 == 0:
                print('Epoch: {} Step: {} Loss: {}'.format(epoch, step, loss))
                
    
tests.test_train_nn(train_nn)




def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)
    print('vgg downloaded')
   

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)
        
        # Build NN using load_vgg, layers, and optimize function
        epochs=10
        batch_size=5
        input_image, keep_prob, layer3_out, layer4_out, layer7_out= load_vgg(sess, vgg_path)
        
        
        layer_out=layers(layer3_out, layer4_out,layer7_out,num_classes)
        label = tf.placeholder(tf.int32, shape=[None, None, None, num_classes])
        learning_rate = tf.placeholder(tf.float32)
        
        
        logits, train_op,loss=optimize(layer_out,label,learning_rate,num_classes)

        # Train NN using the train_nn function        
        train_nn(sess,epochs,batch_size,get_batches_fn,train_op,loss,input_image,label,keep_prob,learning_rate)
        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)



        
if __name__ == '__main__':
    run()
