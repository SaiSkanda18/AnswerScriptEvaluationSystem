from setuptools import setup, find_packages

setup(
    name='automated-answer-evaluation',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='An automated answer script evaluation system integrating NLP and reinforcement learning.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'flask',
        'numpy',
        'pandas',
        'scikit-learn',
        'tensorflow',  # or 'torch' depending on your model
        'opencv-python',  # for OCR
        'pytesseract',  # for OCR
        'nltk',  # for NLP
        'spacy',  # for NLP
        # Add other dependencies as needed
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)