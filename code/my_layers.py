import keras.backend as K
from keras.engine.topology import Layer
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.layers.convolutional import Conv1D
import numpy as np

class Attention(Layer):
    def __init__(self, 
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Content Attention mechanism.
        Supports Masking.
        """
       
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
      
        self.W = self.add_weight((input_shape[-1], ),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((1,),
                                     initializer='zeros',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        else:
            self.b = None


        self.built = True

    def compute_mask(self, input_tensor, mask=None):
        return None

    def call(self, input_tensor, mask=None):
        x = input_tensor
        query = self.W

        query = K.expand_dims(query, axis=-2)
        eij = K.sum(x*query, axis=-1)

        if self.bias:
            eij += self.b

        a = K.exp(eij)
        a_sigmoid = K.sigmoid(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())
            a_sigmoid *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        return [a, a_sigmoid]

   
    def compute_output_shape(self, input_shape):
        return [(input_shape[0], input_shape[1]),
                (input_shape[0], input_shape[1])]




class Self_attention(Layer):
    def __init__(self, use_opinion,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Content Attention mechanism.
        Supports Masking.
        """
       
        self.use_opinion = use_opinion
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(Self_attention, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[0][-1]
        self.steps = input_shape[0][-2]
     
        self.W = self.add_weight((input_dim, input_dim),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_dim,),
                                     initializer='zeros',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, x, mask):
        return mask

    def call(self, input_tensor, mask):
        x = input_tensor[0]
        gold_opinion = input_tensor[1]
        predict_opinion = input_tensor[2]
        gold_prob = input_tensor[3]
        mask = mask[0]
        assert mask is not None

        x_tran = K.dot(x, self.W)
        if self.bias:
            x_tran += self.b 

        x_transpose = K.permute_dimensions(x, (0,2,1))
        weights = K.batch_dot(x_tran, x_transpose)

     
        location = np.abs(np.tile(np.array(range(self.steps)), (self.steps,1)) - np.array(range(self.steps)).reshape(self.steps,1))
        loc_weights = 1.0 / (location+K.epsilon())
        loc_weights *= K.cast((location!=0), K.floatx())
        weights *= loc_weights

        if self.use_opinion:
            gold_opinion_ = gold_opinion[:,:,1]+gold_opinion[:,:,2]
            predict_opinion_ = predict_opinion[:,:,3]+predict_opinion[:,:,4]
            # gold_prob is either 0 or 1 
            opinion_weights = gold_prob*gold_opinion_ + (1.-gold_prob)*predict_opinion_
            opinion_weights = K.expand_dims(opinion_weights, axis=-2)
            weights *= opinion_weights

        weights = K.tanh(weights)
        weights = K.exp(weights)
        weights *= (np.eye(self.steps)==0)

        if mask is not None:
            mask  = K.expand_dims(mask, axis=-2)
            mask = K.repeat_elements(mask, self.steps, axis=1)
            weights *= K.cast(mask, K.floatx())

        weights /= K.cast(K.sum(weights, axis=-1, keepdims=True) + K.epsilon(), K.floatx())

        output = K.batch_dot(weights, x)
        return output


class WeightedSum(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(WeightedSum, self).__init__(**kwargs)

    def call(self, input_tensor, mask=None):
        assert type(input_tensor) == list
        assert type(mask) == list

        x = input_tensor[0]
        a = input_tensor[1]

        a = K.expand_dims(a)
        weighted_input = x * a

        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][-1])

    def compute_mask(self, x, mask=None):
        return None


    
class Conv1DWithMasking(Conv1D):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(Conv1DWithMasking, self).__init__(**kwargs)
    
    def compute_mask(self, x, mask):
        return mask


class Remove_domain_emb(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(Remove_domain_emb, self).__init__(**kwargs)

    def call(self, x, mask=None):
        mask_ = np.ones((400,))
        mask_[300:]=0
        embs = x*mask_
        return embs

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, x, mask):
        return mask


