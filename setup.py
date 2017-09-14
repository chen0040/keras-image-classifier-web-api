from setuptools import setup

setup(
    name='keras_image_classifier_web',
    packages=['keras_image_classifier_web'],
    include_package_data=True,
    install_requires=[
        'flask',
    ],
    setup_requires=[
        'pytest-runner',
    ],
    tests_require=[
        'pytest',
    ],
)