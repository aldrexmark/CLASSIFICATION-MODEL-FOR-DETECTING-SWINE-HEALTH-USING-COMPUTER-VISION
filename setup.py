from setuptools import setup, find_packages

setup(
    name='main',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'Pillow',  # PIL
        'pygame',
        'opencv-python',  # cv2
        'yagmail',
        'ultralytics'
    ],
    entry_points={
        'console_scripts':[
            'startswine=main:main',
        ],
    },
    python_requires='>=3.6',
    include_package_data=True,
    zip_safe=False
)
