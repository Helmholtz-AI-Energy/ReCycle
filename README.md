# Generalized Uncertainty Nonsense (GUN)

Welcome to the RAI-GUN repository. This is mainly a central list for planned features right now

## Datasets

This repository uses a symlink based dataset interface. For this purpose it
includes a directory named `./datasets/`, which is empty by default. Use the
command

`ln -s /path/to/dataset ./datasets/<dataset_name>`

to create a symlink to relevant datasets in this directory. This provides a
homogeneous interface to the datasets on different systems where the datasets
might be saved in different locations. Additionally, this directory easily
transfers to any repository using the same convention.

## Planned features
- [ ] Port Dataset
  - [x] Dumb copy
  - [ ] Review dataset code
  - [ ] Dataset without PCC
- [ ] Linear layer proof of concept
- [ ] Functional model wrapper

with [Pyro](http://pyro.ai) these might be useless or very different
- [ ] Loss with step varying variance
- [ ] Loss with sample dependent variance
- [ ] Loss with fully flexible variance
