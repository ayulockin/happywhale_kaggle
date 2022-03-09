import math
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
import tensorflow_hub as tfhub
import tensorflow_addons as tfa

import efficientnet.tfkeras as efn


class SimpleSupervisedModel():
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
        self.EFNS = [efn.EfficientNetB0, efn.EfficientNetB1, efn.EfficientNetB2, efn.EfficientNetB3, 
                     efn.EfficientNetB4, efn.EfficientNetB5, efn.EfficientNetB6, efn.EfficientNetB7]
        
    def build_model(self):
        if self.args.use_arcface:
            head = ArcMarginProduct
        else:
            assert 1==2, "Turn use_arcface=True"
            
        with self.args.strategy.scope():
            margin = head(
                n_classes = self.args.num_labels, 
                s = 30,
                m = 0.3, 
                name=f'head-arcface', 
                dtype='float32'
            )

            inp = tf.keras.layers.Input(shape=(self.args.image_height, self.args.image_width, 3), name = 'inp1')
            label = tf.keras.layers.Input(shape=(), name = 'inp2')

            x = self.EFNS[self.args.effnet_num](weights='noisy-student', include_top = False)(inp)
            embed = tf.keras.layers.GlobalAveragePooling2D()(x)
            embed = tf.keras.layers.Dropout(0.2)(embed)
            embed = tf.keras.layers.Dense(512)(embed)
            x = margin([embed, label])

            output = tf.keras.layers.Softmax(dtype='float32')(x)

            model = tf.keras.models.Model(inputs = [inp, label], outputs = [output])
            embed_model = tf.keras.models.Model(inputs = inp, outputs = embed)  

            opt = tf.keras.optimizers.Adam(learning_rate=self.args.learning_rate)
            if self.args.freeze_batchnorm:
                self.freeze_batchnorm(model)

            model.compile(
                optimizer = opt,
                loss = [tf.keras.losses.SparseCategoricalCrossentropy()],
                metrics = [tf.keras.metrics.SparseCategoricalAccuracy(),tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)]
                ) 

        return model, embed_model
        
    def freeze_batchnorm(self, model):
        # Unfreeze layers while leaving BatchNorm layers frozen
        for layer in model.layers:
            if not isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = True
            else:
                layer.trainable = False


def get_feature_extractor(model, get_embedding_from='global_average_pooling2d'):     
    feature_extractor = tf.keras.models.Model(
                            model.input,
                            model.get_layer(get_embedding_from).output
                        )
    return feature_extractor
