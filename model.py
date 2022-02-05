import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models


class GetModel():
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
    

def get_feature_extractor(model, get_embedding_from='global_average_pooling2d'):     
    feature_extractor = tf.keras.models.Model(
                            model.input,
                            model.get_layer(get_embedding_from).output
                        )
    return feature_extractor
