# import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D



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
    The builded model.
    """
    model = Sequential()
    input_shape = (width, heigth, 1)

    #flat input 
    model.add(Flatten(input_shape=input_shape))

    #core layers
    for _ in range(depth):
        model.add(Dense(units_x_layer, activation_fc))
        
    #Output 
    model.add(Dropout(dropout_rate))
    model.add(Dense(classes, 'softmax')) # the shape of the output is the desired # of classes
    return model


def simple_cnn(width, heigth, classes):
    model = Sequential()

    model.add(Conv2D(width, (3, 3), activation='relu', input_shape=(width, heigth, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(width*2, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(width*2, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(width*2, activation='relu'))
    model.add(Dropout(0.35))
    model.add(Dense(classes, activation='softmax'))
    return model


def custom_vgg():
    model = Sequential()
    
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(98, 98, 1)))  
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(12, activation='softmax'))
    return model 

    
if __name__ == '__main__':
    #define model parameters

    pass

