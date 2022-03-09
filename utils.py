import os
import re
import wandb
import random
import string
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf


def get_stratified_k_fold(df, target, num_folds):
    """
    Add fold numbers to the given dataframe
    
    Arguments:
    df: Dataframe
    target: List of target to stratify on
    num_folds: Number of folds
    """
    kfold = StratifiedKFold(num_folds, shuffle=True, random_state=42)

    for fold, (train_indices, valid_indices) in enumerate(kfold.split(df, target)):
        df.loc[valid_indices, 'fold'] = int(fold)
        
    return df


class ShowBatch():
    def __init__(self, args):
        self.args = args
        self.id2labels = {val:key for key, val in args.labels.items()}
        
    def get_label_name(self, one_hot_label):
        label = np.argmax(one_hot_label, axis=0)
        return self.id2labels[label]

    def show_batch(self, image_batch, label_batch, type='train_val'):
        plt.figure(figsize=(20,20))
        for n in range(25):
            ax = plt.subplot(5,5,n+1)
            plt.imshow(image_batch[n])
            if type=='train_val':
                plt.title(self.get_label_name(label_batch[n].numpy()))
            plt.axis('off')
            
def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def setup_device():
    "Setup device - GPU or TPU"
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print('Running on TPU ', tpu.master())
    except ValueError:
        tpu = None

    if tpu:
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
        print('#### TPU Available ####')
    else:
        # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
        strategy = tf.distribute.get_strategy()
        if tf.config.list_physical_devices('GPU'):
            print('#### GPU Available ####')
            # TODO - Which GPU?
    
    return strategy

def count_data_items(filenames):
    "Count the number of samples in a TFRecord dataset."
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) 
         for filename in filenames]
    return np.sum(n)


def get_lr_callback(args, plot=False, use_wandb=True):
    lr_start   = 0.000001
    lr_max     = 0.000005 * args.batch_size
    lr_min     = 0.000001
    lr_ramp_ep = 4
    lr_sus_ep  = 0
    lr_decay   = 0.9
   
    def lrfn(epoch):
        if args.resume:
            epoch = epoch + args.resume_epoch
        if epoch < lr_ramp_ep:
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
        elif epoch < lr_ramp_ep + lr_sus_ep:
            lr = lr_max
        else:
            lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min
            
        # log the current learning rate onto W&B
        if use_wandb:
            if wandb.run is None:
                raise wandb.Error("You must call wandb.init() before WandbCallback()")

            wandb.log({'learning_rate': lr}, commit=False)
            
        return lr
        
    if plot:
        epochs = list(range(args.epochs))
        learning_rates = [lrfn(x) for x in epochs]
        plt.scatter(epochs,learning_rates)
        plt.show()

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=False)
    return lr_callback

def model_ckpt_callback(args, fold):
    os.makedirs(f'{args.model_save_path}/{args.exp_id}', exist_ok=True)
    
    return tf.keras.callbacks.ModelCheckpoint(
        f'{args.model_save_path}/{args.exp_id}_{fold}',
        monitor='val_loss',
        verbose=0,
        save_best_only=True,
        save_weights_only=False,
        mode='min',
        save_freq='epoch'
    )
