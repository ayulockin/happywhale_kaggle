import numpy as np
import tensorflow as tf
from functools import partial
# Imports for augmentations. 
from albumentations import Compose, RandomResizedCrop, Cutout, Rotate, HorizontalFlip, VerticalFlip,\
                           RandomBrightnessContrast, ShiftScaleRotate, CenterCrop, Resize, Normalize

AUTOTUNE = tf.data.AUTOTUNE


class GetDataloader():
    def __init__(self, args):
        self.args = args
        
    def dataloader(self, df, data_type='train'):
        '''
        Return train, validation or test dataloader
        
        Args:
            df: Pandas dataframe
            data_type: Anyone of one train, valid, or test.
        '''
        # Consume dataframe
        dataloader = tf.data.Dataset.from_tensor_slices(dict(df))
        
        # SHuffle if its for training
        if data_type=='train':
            dataloader = dataloader.shuffle(self.args.batch_size)

        # Load the image
        dataloader = (
            dataloader
            .map(partial(self.parse_data, data_type=data_type), num_parallel_calls=AUTOTUNE)
            .cache()
        )
        
        # Add augmentation to dataloader for training
        if self.args.use_augmentations and data_type=='train':
            self.transform = self.build_augmentation(data_type=data_type)
            dataloader = dataloader.map(self.augmentation, num_parallel_calls=AUTOTUNE)
            
        # Add general stuff
        dataloader = (
            dataloader
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
    def parse_data(self, df_dict, data_type='train'):
        # Parse Image
        image = tf.io.read_file(df_dict['img_path'])
        image = self.decode_image(image)

        if data_type in ['train', 'valid']:
            # Parse Target
            label = df_dict['target']
            if self.args.use_arcface:
                label = tf.cast(label, tf.int32)
            else:
                label = tf.one_hot(indices=label, depth=self.args.num_labels)
            
            if self.args.use_arcface:
                return {'img_input': image, 'label_input': label}, label

            return image, label
        
        elif data_type == 'test':
            return image
        
        else:
            raise NotImplementedError("Not implemented for this data_type")
            
    def build_augmentation(self, data_type='train'):
        if data_type=='train':
            transform = Compose([
                CenterCrop(90, 90, p=0.5),
                Resize(self.args.image_height, self.args.image_width, p=1),
                Rotate(limit=10),
                Cutout(num_holes=8, max_h_size=10, max_w_size=10, p=1.0),
                HorizontalFlip(p=0.7),

                # VerticalFlip(p=0.4),
                # Normalize(
                #     mean=[0.485, 0.456, 0.406],
                #     std=[0.229, 0.224, 0.225],
                #     max_pixel_value=255.0,
                #     p=1.0,
                # ),
            ])

        elif data_type=='valid':
            transform = Compose([
                # Resize(CONFIG['img_height'], CONFIG['img_width'], p=1),
                Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                    p=1.0,
                ),
            ])
        else:
            raise NotImplementedError("No augmentation available for this data_type")

        return transform
            
    def augmentation(self, image, label):
        aug_img = tf.numpy_function(func=self.aug_fn, inp=[image], Tout=tf.float32)
        aug_img.set_shape((self.args.image_height, self.args.image_width, 3))

        return aug_img, label

    def aug_fn(self, image):
        data = {"image":image}
        aug_data = self.transform(**data)
        aug_img = aug_data["image"]

        return aug_img.astype(np.float32) 
