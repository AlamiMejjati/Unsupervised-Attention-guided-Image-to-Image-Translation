"""Contains the standard train/test splits for the cyclegan data."""

"""The size of each dataset. Usually it is the maximum number of images from
each domain."""
DATASET_TO_SIZES = {
    'horse2zebra_train': 1334,
    'horse2zebra_test': 140,
    'apple2orange_train': 1019,
    'apple2orange_test': 266,
    'antelope2deer_train': 1217,
    'antelope2deer_test': 127,
    'german2collie_train': 929,
    'german2collie_test': 104,
    'lion2tiger_train': 916,
    'lion2tiger_test': 103,
    'summer2winter_yosemite_train': 1231,
    'summer2winter_yosemite_test': 309,
}

"""The image types of each dataset. Currently only supports .jpg or .png"""
DATASET_TO_IMAGETYPE = {
    'horse2zebra_train': '.jpg',
    'horse2zebra_test': '.jpg',
    'apple2orange_train': '.jpg',
    'apple2orange_test': '.jpg',
    'antelope2deer_train': '.jpg',
    'antelope2deer_test': '.jpg',
    'german2collie_train': '.jpg',
    'german2collie_test': '.jpg',
    'lion2tiger_train': '.jpg',
    'lion2tiger_test': '.jpg',
    'summer2winter_yosemite_train': '.jpg',
    'summer2winter_yosemite_test': '.jpg',
}

"""The path to the output csv file."""
PATH_TO_CSV = {
    'horse2zebra_train': './input/horse2zebra/horse2zebra_train.csv',
    'horse2zebra_test': './input/horse2zebra/horse2zebra_test.csv',
    'apple2orange_train': './input/apple2orange/apple2orange_train.csv',
    'apple2orange_test': './input/apple2orange/apple2orange_test.csv',
    'antelope2deer_train': './input/antelope2deer/antelope2deer_train.csv',
    'antelope2deer_test': './input/antelope2deer/antelope2deer_test.csv',
    'german2collie_train': './input/german2collie/german2collie_train.csv',
    'german2collie_test': './input/german2collie/german2collie_test.csv',
    'lion2tiger_train': './input/lion2tiger/lion2tiger_train.csv',
    'lion2tiger_test': './input/lion2tiger/lion2tiger_test.csv',
    'summer2winter_yosemite_train': './input/summer2winter_yosemite/summer2winter_yosemite_train.csv',
    'summer2winter_yosemite_test': './input/summer2winter_yosemite/summer2winter_yosemite_test.csv'
}
