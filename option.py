'''
    Options for dwsr-pytorch.
'''

opt = {
    # dataset
    'train_dir': r'path_to_your_dataset/train',
    'val_dir': r'path_to_your_dataset/val',
    'test_dir': r'path_to_your_dataset/test',
    'crop_size': 128,
    'num_workers': 16,
    # model settings
    'n_conv': 8,
    'residue_weight': 1,
    # training settings
    'epochs': 100,
    'batch_size': 64,
    'lr': 1e-2,
    # test settings
    'test_ckp': './out/ckp/epoch_100.pth'
}
