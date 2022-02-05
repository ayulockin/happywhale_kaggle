import tensorflow as tf
from functools import partial

AUTOTUNE = tf.data.AUTOTUNE

# def get_dataloader(df, args, type='train'):
#     resize = args.resize
#     image_height = args.image_height
#     image_width = args.image_width
    
#     dataloader = tf.data.Dataset.from_tensor_slices(dict(df))
    
#     if type=='train':
#         dataloader = dataloader.shuffle(args.batch_size)
        
#     dataloader = (
#         dataloader
#         # .map(partial(parse_data, resize=resize, img_height=image_height, img_width=image_width),
#         #      num_parallel_calls=AUTOTUNE)
#         .map(partial(parse_data, args),
#              num_parallel_calls=AUTOTUNE)
#         .cache()
#         .batch(args.batch_size)
#         .prefetch(AUTOTUNE)
#     )

#     return dataloader


class GetDataloader():
    def __init__(self, args):
        self.args = args
        
    def dataloader(self, df, data_type='train'):
        dataloader = tf.data.Dataset.from_tensor_slices(dict(df))
    
        if data_type=='train':
            dataloader = dataloader.shuffle(self.args.batch_size)

        dataloader = (
            dataloader
            .map(partial(self.parse_data, type='train_val'), num_parallel_calls=AUTOTUNE)
            .cache()
            .batch(self.args.batch_size)
            .prefetch(AUTOTUNE)
        )

        return dataloader
        
    @tf.function
    def decode_image(self, img):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=3)
        # Normalize image
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)
        # resize the image to the desired size
        if self.args.resize:
            img = tf.image.resize(img, [self.args.image_height, self.args.image_width], 
                                  method='bicubic', preserve_aspect_ratio=False)
            img = tf.clip_by_value(img, 0.0, 1.0)

        return img

    @tf.function
    def parse_data(self, df_dict, type='train_val'):
        # Parse Image
        image = tf.io.read_file(df_dict['img_path'])
        image = self.decode_image(image)

        if type=='train_val':
            # Parse Target
            label = df_dict['target']
            label = tf.one_hot(indices=label, depth=self.args.num_labels)

            return image, label

        return image  