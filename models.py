# import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

from testing import *


# infor ref de como construir modeles en tensorflow
#  https://www.pyimagesearch.com/2019/10/28/3-ways-to-create-a-keras-model-with-tensorflow-2-0-sequential-functional-and-model-subclassing/

def simple_stacked_fc_nn(width, heigth, depth, units_x_layer, classes, dropout_rate=0.2, activation_fc='relu'):
    """Builds a fully connected nn of n depth with intermidiate drops layers

    Here's the layout of the nn:

     (input) <-- input is 2d (width, heigth, 1)
        v
    [Flatten]
        v
    [fully connected layer] <-- units_x_layer nodes
        v
    [DropOut Layer] <-- dropout_rate
        v
        .
        . <-- Repeat n times (depth) the same fully connected - dropOut structure
        .
        v
    [fully connected layer]
        v
    (output)
        
    Args:
    width:
    heigth, 
    layers, 
    classes, 
    dropout_rate
    activation_fc

    Returns:
    the builded model.
    """
    model = Sequential()
    input_shape = (width, heigth)

    #flat input 
    model.add(Flatten(input_shape=input_shape))

    #core layers
    for _ in range(depth):
        model.add(Dense(units_x_layer, activation_fc))
        model.add(Dropout(dropout_rate))
        
    #Output 
    model.add(Dense(classes, 'softmax')) # the shape of the output is the desired # of classes
    return model



if __name__ == '__main__':
    model = simple_stacked_fc_nn(224,
                                 224,
                                 1,
                                 512,
                                 12)
    print(model.summary())
    train_ds = get_dataset_from_tfrecords()
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_ds, epochs=2)