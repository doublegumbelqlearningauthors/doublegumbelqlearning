# Double Gumbel Q-Learning (DoubleGum)

## Installation

On `Python 3.7` with `Cuda 11.1.1` and `cudnn 8.2.0`.

create virtualenv
```
virtualenv <VIRTUALENV_LOCATION>/doublegum
source <VIRTUALENV_LOCATION>/doublegum
```

within `<REPOSITORY_LOCATION>` install from requirements
```
git clone https://github.com/doublegumbelqlearningauthors/doublegumbelqlearning.git
cd doublegumbelqlearning
pip install -r requirements.txt
```

install mujoco
```commandline
mkdir .mujoco
cd .mujoco
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
tar -xf mujoco210-linux-x86_64.tar.gz
```

## Continuous Control

```commandline
main_cont.py --env <ENV_NAME> --policy <DoubleGum/DDPG/TD3/MoG-DDPG>
```
MetaWorld `env`s are run with `--env MetaWorld_<ENVNAME>`

## Discrete Control

```commandline
main_disc.py --env <ENV_NAME> --policy <DoubleGum/DQN/DuellingDDQN>
```

## Plotting

```commandline
cd <REPOSITORY_LOCATION>/doublegumbelqlearning/
cd ..
wget https://drive.google.com/file/d/1Wl1fSi3bm3pcSkbeMMcbv0VvGc4nHBzw/view?usp=sharing
uzip doublegumbelqlearning-results.zip
cd doublegumbelqlearning
```

Individual runs may be plotted by
```commandline
python plotting/<SCRIPT_NAME>.py
```
