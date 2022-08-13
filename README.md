# Overview

This respository is a PyTorch implementation of Multi-Agent Reinforcement Learning (MARL) algorithms including:
* [VDN](https://arxiv.org/abs/1706.05296)
* [QMIX](https://arxiv.org/abs/1803.11485)
* [QTRAN](https://arxiv.org/abs/1905.05408)
* [Qatten](https://arxiv.org/abs/2002.03939)
* [COMA](https://arxiv.org/abs/1705.08926)

The running environment is [SMAC](https://github.com/oxwhirl/smac). This implementation supports parallel runners in experiment.

# Setup

Set up the working environment:

```shell
pip install -r requirements.txt 
```

Set up the StarCraftII game core

```shell
bash install_sc2.sh  
```

# Run

For example, to run `QMIX` on the map `3m`:

```shell
python main.py --alg=qmix --map=3m
```

Running results are saved under the folder:

`./results/`

Trained models are saved under the folder:

`./models/`

More information on configuration can be found in:

`./common/arguments.py`

## Licence

The MIT License