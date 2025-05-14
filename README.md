# Installation

## 1. Install dependencies

```bash
conda env create --name envname --file=environments.yml
```

## 2. Download models
```bash
cd /tmp
git clone https://github.com/Gepetto/example-robot-data/
cd THIS_REPO_FOLDER
cp -r /tmp/example-robot-data/robots ./models/example-robot-data/
```

# How to run

python meshcat_viewer.py

# Useful links

How to install: https://stack-of-tasks.github.io/pinocchio/download.html

Docs: https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/master/doxygen-html/

Viewer example: https://github.com/stack-of-tasks/pinocchio/blob/master/examples/meshcat-viewer.py

Models: https://github.com/stack-of-tasks/pinocchio/blob/master/models/

Tutorial: https://github.com/rocketman123456/pinocchio_tutorial

# Useful commands

- Save conda env

```bash
conda env export > environment.yml
conda list --export > requirements.txt
```
