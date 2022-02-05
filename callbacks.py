import tensorflow as tf

class GetCallbacks():
    def __init__(self, args):
        self.args = args
        
    def get_earlystopper(self):
        earlystopper = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=self.args.early_patience, verbose=0, mode='auto',
            restore_best_weights=True
        )
        
        return earlystopper

    def get_reduce_lr_on_plateau(self):
        reduce_lr_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=self.args.rlrp_factor, patience=self.args.rlrp_patience
        )
        
        return reduce_lr_on_plateau
