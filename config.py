from argparse import Namespace

LABEL2IDS = {
    'melon_headed_whale': 0,
    'humpback_whale': 1,
    'false_killer_whale': 2,
    'bottlenose_dolphin': 3,
    'beluga': 4,
    'minke_whale': 5,
    'fin_whale': 6,
    'blue_whale': 7,
    'gray_whale': 8,
    'southern_right_whale': 9,
    'common_dolphin': 10,
    'killer_whale': 11,
    'short_finned_pilot_whale': 12,
    'dusky_dolphin': 13,
    'long_finned_pilot_whale': 14,
    'sei_whale': 15,
    'spinner_dolphin': 16,
    'cuviers_beaked_whale': 17,
    'spotted_dolphin': 18,
    'brydes_whale': 19,
    'commersons_dolphin': 20,
    'white_sided_dolphin': 21,
    'rough_toothed_dolphin': 22,
    'pantropic_spotted_dolphin': 23,
    'pygmy_killer_whale': 24,
    'frasiers_dolphin': 25
}

train_config = Namespace(
    # DATA
    train_img_path = '../128x128/train_images-128-128/train_images-128-128',
    labels = LABEL2IDS,
    num_labels = len(LABEL2IDS),
    image_height = 128,
    image_width = 128,
    resize=False,
    
    # CROSS VALIDATION
    num_folds = 5,
    
    # TRAIN
    batch_size = 256, # can I try 256
    epochs = 30, # default 30
    early_patience = 6,
    rlrp_factor = 0.2,
    rlrp_patience = 3,
    
    # MODEL
    model_save_path = '../models',
    
    # EMBEDDING
    embedding_save_path = '../embeddings',
    
    # ARCFACE
    use_arcface = False,
    
    # Augmentation
    use_augmentations = True
)

test_config = Namespace(
    # DATA
    test_img_path = '../128x128/test_images-128-128/test_images-128-128',
    labels = LABEL2IDS,
    num_labels = len(LABEL2IDS),
    image_height = 128,
    image_width = 128,
    resize=False,
    batch_size = 256, # can I try 256
    
    # CROSS VALIDATION
    num_folds = 5,
    
    # MODEL
    model_save_path = '../models',
    
    # EMBEDDING
    embedding_save_path = '../embeddings'
) 
    

def get_train_config():
    return train_config

def get_test_config():
    return test_config
