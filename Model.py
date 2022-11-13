import numpy as np
# import pandas as pd
from keras.layers import Convolution1D, Add, Concatenate, MaxPooling1D, MultiHeadAttention, Dropout, LayerNormalization, Dense, Flatten, GlobalAveragePooling1D, ReLU
from keras.activations import sigmoid
from keras.models import Model
from keras import Input, optimizers
import tensorflow as tf

#create TRANSFORMER
def add_position(input,num_lead):
    input_pos_encoding = tf.constant(num_lead, shape=[input.shape[1]], dtype="int32")/input.shape[1]
    input_pos_encoding = tf.cast(tf.reshape(input_pos_encoding, [1,10]),tf.float32)
    input = tf.add(input ,input_pos_encoding)
    return input
def transformer_encoder(input,key_dim,num_heads,dropout):
    # Normalization and Attention
    x = LayerNormalization(epsilon=1e-6)(input)
    x = MultiHeadAttention(
        key_dim=key_dim, num_heads=num_heads
    )(x, x)
    x = Dropout(dropout)(x)
    res = x + input

    # Feed Forward Part
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(key_dim, activation='relu')(x)
    return x + res
def stack_block_transformer(key_dim,num_transformer_blocks,dropout):
    input = Input(shape=(1, key_dim))
    x = input
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x,key_dim,num_heads = 2,dropout = dropout)
    return input, x

#create SE-RESNET
def SE_block(input,in_channel,reduction,downsample = False):
    x = input
    if downsample: 
        #se_branch
        x = Convolution1D(in_channel,7,2,padding = "same")(x)
        x = Convolution1D(in_channel,7,1,padding = "same")(x)
        x = SE_reduction(x,in_channel,reduction)
        #residual branch
        input = Convolution1D(in_channel,1,2,padding = "valid")(input)
    else:
        #se_branch
        x = Convolution1D(in_channel,7,1,padding = "same")(x)
        x = Convolution1D(in_channel,7,1,padding = "same")(x)
        x = SE_reduction(x,in_channel,reduction)
        #nothing in residual branch
    return input + x    
def SE_reduction(input,in_channel,reduction):
    attention_score = GlobalAveragePooling1D(keepdims = True)(input)
    #squeeze
    attention_score = Convolution1D(int(in_channel/reduction),1)(attention_score) #padding = same, stride = 1
    attention_score = ReLU()(attention_score)
    #excitement
    attention_score = Convolution1D(in_channel,1)(attention_score)
    #multiply score
    return input*sigmoid(attention_score)
def SE_ResNet(input):
    x = Convolution1D(64,15,2,padding="valid")(input)
    x = MaxPooling1D(3,2,padding="valid")(x)
    #stage1
    x = SE_block(x,64,16)
    #stage2
    x = SE_block(x,128,16,True)
    #stage3
    x = SE_block(x,256,16,True)
    #stage4
    x = SE_block(x,512,16,True)
    x = GlobalAveragePooling1D(keepdims = True)(x)
    #x = Dense(SLE_dim,activation = "relu")(x)
    return x

#create MODEL
def Proposed_model(key_dim,num_lead,num_class):
    #hyperparameter
    num_transformer_blocks = 2
    #initialize first input
    input_, transformer_ = stack_block_transformer(key_dim,num_transformer_blocks,0,1)
    inputs = []
    transformers = []
    transformers.append(transformer_)
    inputs.append(input_)
    #stack transformers
    for i in range(1,num_lead):
        input_i, transformer_i = stack_block_transformer(num_transformer_blocks)
        inputs.append(input_i) 
        transformer_i = add_position(transformer_i,i)
        transformers.append(transformer_i)
    #spatial feature
    x = Concatenate(transformers, axis=-1)
    x = tf.expand_dims(x, axis = 0)
    feature = SE_ResNet(x)

    #metadata encoding 
    metadata = Input(shape= feature.shape)

    #attention architecture
    feature1 = MultiHeadAttention(num_heads=2, key_dim= 512)(metadata,feature)
    feature1 = Add()([feature,feature1])
    feature2 = MultiHeadAttention(num_heads=2, key_dim= 512)(feature,metadata)
    feature2 = Add()([metadata,feature2])
    feature = Concatenate()([feature1,feature2])
    #classify
    feature = LayerNormalization(epsilon=1e-6)(feature)
    feature = Dense(128,activation="relu")(feature)
    feature = Dropout(0.1)(feature)
    prediction = Dense(num_class,activation="sigmoid")(feature)
    #optimizer
    model = Model([inputs,metadata],prediction)
    opt = optimizers.Adagrad(learning_rate=0.001)
    model.compile(
        loss="binary_crossentropy",
        optimizer=opt,
        metrics=["accuracy"]
        )
    print(model.summary)
    return model
