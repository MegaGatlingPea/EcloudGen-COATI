# Installation

Before installation, we recommend to first setup a Python virtual environment.

```bash
$ pip install .
```

# PART 1 

Preprocess the training data from SMILES and ecloud data (change the file name to your own data). 

```
$ python part1_data_prepare.py
```


# PART 2 

Train the model using DDP. 
```
$ torchrun --standalone part2_train_grande.py
```

# PART 3

Test the performance of the trained model. We provide four case modes here: `basic`, `near`, `regression`, `dynamics`. 
```
$ python -u examples/gen.py --mode <MODE>
```

# Models 

Trained model is provided in `models/ecloud_augmented_37.pkl`. 
