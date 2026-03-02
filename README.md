# Get Started with TokaMark

## Environment Setup

Create and activate the required Conda environment and install the preprocessing package:

```bash
git clone git@github.ibm.com:rousseau-cecile/fairmast-data-preprocessing.git
cd fairmast-data-preprocessing
conda env create -f environment_basic.yml
conda activate myenv
pip install -e .
```

## Training and Evaluation
To run training and evaluation on a small subset of the MAST dataset for quick testing:

```bash
cd cnn-baseline
python run_training.py --task task_1-1
python run_evaluation.py --task task_1-1
```
