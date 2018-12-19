# keras_shufflenetv2

ShufflenetV2
============

Layer (type) | Output Shape | Param #
------------ | ------------ | -------
conv2d (Conv2D) | multiple         |         640       
conv2d_1 (Conv2D)           | multiple          |        1056      
separable_conv2d (SeparableC| multiple          |        1344      
max_pooling2d (MaxPooling2D)| multiple          |        0         
dense (Dense)               | multiple          |        6423040   
dense_1 (Dense)             | multiple          |        5130      
batch_normalization (BatchNo| multiple          |        128       
flatten (Flatten)           | multiple          |        0         
dropout (Dropout)           | multiple          |        0         


epoch = 200
-----------
acc = 94.57
-----------


CNN
===


Layer (type)       |          Output Shape        |      Param #   
------------ | ------------ | -------
conv2d_2 (Conv2D)    |        multiple          |        320       
conv2d_3 (Conv2D)        |    multiple          |        18496     
max_pooling2d_1 (MaxPooling2 |multiple          |        0         
dense_2 (Dense)             | multiple          |        6423040   
dense_3 (Dense)             | multiple          |        5130      
flatten_1 (Flatten)         | multiple          |        0         
dropout_1 (Dropout)         | multiple          |        0         


epoch = 200
-----------
acc = 84.75
-----------
