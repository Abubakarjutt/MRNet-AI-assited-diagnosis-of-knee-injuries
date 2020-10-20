# AI assited medical diagnosis of knee injuries.

This repository contains an implementation of research paper <a href="https://stanfordmlgroup.github.io/projects/mrnet/">MRNet: Deep-learning-assisted diagnosis for knee magnetic resonance imaging</a> with a slight modification.



## Dataset: MRNet 

The data comes from Stanford ML Group research lab. It consits of 1,370 knee MRI exams performed at Stanford University Medical Center to study the presence of Anterior Cruciate Ligament (ACL) and Meniscus tears. The problem is a multilabe multiclass problem and the input data is multi axis (sagittal, coronal, axial).


## How to use the code:

You can request this dataset from this <a href="https://stanfordmlgroup.github.io/competitions/mrnet/">link</a>.

inside the downloaded data folder there will be two folders `train` and `valid` containing MRI scans and their labels as seperate csv files.

You can execute it with the following arguments:

```python
parser = argparse.ArgumentParser()
parser.add_argument('--augment', type=int, choices=[0, 1], default=1)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--flush_history', type=int, choices=[0, 1], default=0)
parser.add_argument('--save_model', type=int, choices=[0, 1], default=1)
parser.add_argument('--patience', type=int, choices=[0, 1], default=20)
```

example to train a model to detect acl tears on the sagittal plane for a 20 epochs:

`python train.py --epochs=20`

Note: Before running the script, add the following (empty) folders at the root of the project:
- models
- logs


## Results:

I trained this end to end multilabel multiclaass model and got following AUC scores:

- on train: 0.99
- on validation: 0.915




## Contributions - PR are welcome:
If you feel that some functionalities or improvements could be added to the project, don't hesitate to submit a pull request.

