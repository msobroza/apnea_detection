from keras.layers import (Input, Dense, BatchNormalization, Dropout, Activation, Concatenate, Lambda, Flatten)
from keras.models import Model, Sequential
import keras.backend as K

    
def attention_pooling(inputs, **kwargs):
    [out, att]      = inputs
    epsilon         = 1e-7
    att             = K.clip(att, epsilon, 1. - epsilon)
    normalized_att  = att / K.sum(att, axis=1)[:, None, :]

    return K.sum(out * normalized_att, axis=1)
        
def pooling_shape(input_shape):
    if isinstance(input_shape, list):
        (sample_num, _, freq_bins) = input_shape[0]
    else:
        (sample_num, _, freq_bins) = input_shape
    
    return (sample_num, freq_bins)

def DLMA(
    time_steps=10,
    freq_bins=128,
    classes_num=527,
    hidden_units=1024,
    drop_rate=0.5
):
    '''Decision Multi Level Attention'''

    # Embedded layers
    input_layer = Input(shape=(time_steps, freq_bins))

    a1 = Dense(hidden_units)(input_layer)
    a1 = BatchNormalization()(a1)
    a1 = Activation('relu')(a1)
    a1 = Dropout(drop_rate)(a1)

    a2 = Dense(hidden_units)(a1)
    a2 = BatchNormalization()(a2)
    a2 = Activation('relu')(a2)
    a2 = Dropout(drop_rate)(a2)

    a3 = Dense(hidden_units)(a2)
    a3 = BatchNormalization()(a3)
    a3 = Activation('relu')(a3)
    a3 = Dropout(drop_rate)(a3)

    # Pooling layers
    cla1 = Dense(classes_num, activation='sigmoid')(a2)
    att1 = Dense(classes_num, activation='softmax')(a2)
    out1 = Lambda(attention_pooling, output_shape=pooling_shape)([cla1, att1])

    cla2 = Dense(classes_num, activation='sigmoid')(a3)
    att2 = Dense(classes_num, activation='softmax')(a3)
    out2 = Lambda(attention_pooling, output_shape=pooling_shape)([cla2, att2])

    b1 = Concatenate(axis=-1)([out1, out2])
    b1 = Dense(classes_num)(b1)

    output_layer = Activation('softmax')(b1)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model


def SBL(
    time_steps=3,
    freq_bins=128,
    hidden_units_rnn=256,
    hidden_units=1024,
    batch_size=500
):
    '''Simple Binary Classifier'''

    model = Sequential()

    model.add(Flatten())
    model.add(Dense(freq_bins))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    return model