import math
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models


class SimpleSupervisedtModel():
    def __init__(self, args):
        self.args = args
        
    def get_efficientnet(self):
        """
        Get baseline efficientnet model
        """
        base_model = tf.keras.applications.EfficientNetB0(include_top=False, weights='imagenet')
        base_model.trainabe = True

        inputs = layers.Input((self.args.image_height, self.args.image_width, 3))
        x = base_model(inputs, training=True)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(self.args.num_labels, activation='softmax')(x)

        return models.Model(inputs, outputs)
    
    
# Arcmarginproduct class keras layer
class ArcMarginProduct(tf.keras.layers.Layer):
    '''
    Implements large margin arc distance.

    Reference:
        https://arxiv.org/pdf/1801.07698.pdf
        https://github.com/lyakaap/Landmark2019-1st-and-3rd-Place-Solution/
            blob/master/src/modeling/metric_learning.py
    '''
    def __init__(self, n_classes, s=30, m=0.50, easy_margin=False,
                 ls_eps=0.0, **kwargs):

        super(ArcMarginProduct, self).__init__(**kwargs)

        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.ls_eps = ls_eps
        self.easy_margin = easy_margin
        self.cos_m = tf.math.cos(m)
        self.sin_m = tf.math.sin(m)
        self.th = tf.math.cos(math.pi - m)
        self.mm = tf.math.sin(math.pi - m) * m

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'n_classes': self.n_classes,
            's': self.s,
            'm': self.m,
            'ls_eps': self.ls_eps,
            'easy_margin': self.easy_margin,
        })
        return config

    def build(self, input_shape):
        super(ArcMarginProduct, self).build(input_shape[0])

        self.W = self.add_weight(
            name='W',
            shape=(int(input_shape[0][-1]), self.n_classes),
            initializer='glorot_uniform',
            dtype='float32',
            trainable=True,
            regularizer=None)

    def call(self, inputs):
        X, y = inputs
        y = tf.cast(y, dtype=tf.int32)
        cosine = tf.matmul(
            tf.math.l2_normalize(X, axis=1),
            tf.math.l2_normalize(self.W, axis=0)
        )
        sine = tf.math.sqrt(1.0 - tf.math.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = tf.where(cosine > 0, phi, cosine)
        else:
            phi = tf.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = tf.cast(
            tf.one_hot(y, depth=self.n_classes),
            dtype=cosine.dtype
        )
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.n_classes

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output

    
class ArcFaceSupervisedModel():
    def __init__(self, args):
        self.args = args
        
    def get_efficientnet(self):
        """
        Get arcface based efficientnet model
        """
        # Base model
        base_model = tf.keras.applications.EfficientNetB0(include_top=False, weights='imagenet')
        base_model.trainabe = True

        # Initialize ArcFace layer
        margin = ArcMarginProduct(
            n_classes = self.args.num_labels, 
            s = 30, 
            m = 0.3, 
            name='arcface', 
            dtype='float32'
        )
        
        img_inputs = layers.Input((self.args.image_height, self.args.image_width, 3), name='img_input')
        label_inputs = layers.Input(shape=(), name='label_input')
        
        x = base_model(img_inputs, training=True)
        embed = layers.GlobalAveragePooling2D()(x)
        x = margin([embed, label_inputs])
        
        x = layers.Dropout(0.5)(x)
        output = layers.Softmax(dtype='float32')(x)
        
        return models.Model(inputs=[img_inputs, label_inputs], outputs=[output])


def get_feature_extractor(model, get_embedding_from='global_average_pooling2d'):     
    feature_extractor = tf.keras.models.Model(
                            model.input,
                            model.get_layer(get_embedding_from).output
                        )
    return feature_extractor
