# ReCycle - Residual Cyclic Time-Series Forecasting

Welcome to the RAI-ReCycle repository. This is a stub an will be expanded in the future

## Datasets

This repository uses a symlink based dataset interface. For this purpose it
includes a directory named `./datasets/`, which is empty by default. Use the
command

`ln -s /path/to/dataset ./datasets/<dataset_name>`

to create a symlink to relevant datasets in this directory. This provides a
homogeneous interface to the datasets on different systems where the datasets
might be saved in different locations. Additionally, this directory easily
transfers to any repository using the same convention. As side effect all code
in this repository needs to be run from its main directory for the relative 
paths to work.

This is a feature for development.

Naming conventions can be found in `data/datset_bib.py`


## Planned features
 - [ ] Dataframe interface
 - [ ] Metadata flexibility
