import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold


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
        df.loc[valid_indices, 'fold'] = fold
        
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
