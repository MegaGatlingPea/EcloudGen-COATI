import os
import glob
all=glob.glob("./examples/conditions/target_ic50/*.txt")
for file in all:
    command = "python -u gen.py --mode regression --regression_y pic50 --smiles " + file
    print (command)
    os.system(command)