from setuptools import setup

setup(
    name='keras_image_classifier',
    packages=['keras_image_classifier'],
    include_package_data=True,
    install_requires=[
        'flask',
        'keras',
        'tensorflow',
        'pillow',
        'numpy',
        'h5py',
        'scikit-learn'
    ],
    setup_requires=[
        'pytest-runner',
    ],
    tests_require=[
        'pytest',
    ],
)