from keras.layers import Input, merge
import keras.backend  as K
from keras.layers.core import Layer, Reshape
from keras.models import Model
import numpy as np

class BatchBatchDot(Layer):
    def __init__(self, shape1, shape2, dot_axe = 2, **kwargs):
        super(BatchBatchDot, self).__init__(**kwargs)
        assert shape1[0] == shape2[0]
        self.shape1 = tuple(shape1)
        self.shape2 = tuple(shape2)
        self.dot_axe = dot_axe
        self.batch_output_shape = self.get_output_shape_for([(-1,) + shape1, (-1,) + shape2])
        


    def get_output_shape_for(self, input_shape):
        input_shapes = input_shape
        shape1 = list(input_shapes[0])
        shape2 = list(input_shapes[1])
        shape1.pop(self.dot_axe)
        shape2.pop(self.dot_axe)
        shape2.pop(0)
        shape2.pop(0)
        output_shape = shape1 + shape2
        if len(output_shape) == 1:
            output_shape += [1]
        return tuple(output_shape)

    def call(self, inputs, mask=None):
        l1 = inputs[0]
        l2 = inputs[1]
        l1 = K.reshape(l1, (-1, ) + self.shape1[1:])
        l2 = K.reshape(l2, (-1, ) + self.shape2[1:])
        output = K.batch_dot(l1, l2, self.dot_axe-1)
        output = K.reshape(output, self.batch_output_shape)
        return output

    def get_config(self):
        config = {'shape1': self.shape1, 'shape2': self.shape2}
        base_config = super(BatchBatchDot, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


input1 = Input(shape=(4,3))
input2 = Input(shape=(4,3,5))
# input1_reshape = BatchReshape(target_shape = (3, ))
# input2_reshape = BatchReshape(target_shape = (3, 5))
# _input1 = input1_reshape(input1)
# _input2 = input2_reshape(input2)
# print input1_reshape.output_shape
# print input2_reshape.output_shape
# out_layer = Dot(2)
# output = out_layer([input1, input2])
# output = merge(inputs = [_input1, _input2], dot_axes = 1, mode = 'dot')
output_layer = BatchBatchDot((4, 3), (4, 3, 5))
output = output_layer([input1, input2])
model = Model(input=[input1, input2], output=output)
a = np.ones((2,4,3))
b = np.ones((2,4,3,5))
c = (model.predict([a,b]))
print output_layer.output_shape
print output_layer.batch_output_shape
print(c.shape)
print(c)