import re
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