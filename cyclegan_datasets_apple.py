"""Contains the standard train/test splits for the cyclegan data."""

"""The size of each dataset. Usually it is the maximum number of images from
each domain."""
DATASET_TO_SIZES = {
    'apple2orange_train': 1019,
    'apple2orange_test': 266
}

"""The image types of each dataset. Currently only supports .jpg or .png"""
DATASET_TO_IMAGETYPE = {
    'apple2orange_train': '.jpg',
    'apple2orange_test': '.jpg',
}

"""The path to the output csv file."""
PATH_TO_CSV = {
    'apple2orange_train': './input/apple2orange/apple2orange_train.csv',
    'apple2orange_test': './input/apple2orange/apple2orange_test.csv',
}
