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

# Useful links

Docs: https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/master/doxygen-html/

Viewer example: https://github.com/stack-of-tasks/pinocchio/blob/master/examples/meshcat-viewer.py

Models: https://github.com/stack-of-tasks/pinocchio/blob/master/models/
