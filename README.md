# RLDemo
Two kinds of model-free Reinforcement Learning methods, value-based RL and policy-based RL, are implemented respectively to solve two kinds of environments, with discrete action space or continuous action space. Each kind of method is implemented with a basic and popular algorithm and its corresponding representative improvement. And each kind of environment is tested with four different instances. Specifically, [**Deep Q-Network**](https://arxiv.org/abs/1312.5602) ([**DQN**](https://www.nature.com/articles/nature14236)) with [**Dueling Double DQN**](https://arxiv.org/abs/1511.06581) ([**D3QN**](https://arxiv.org/abs/1509.06461)) and [**Deep Deterministic Policy Gradient**](https://arxiv.org/abs/1509.02971) (**DDPG**) with [**Twin Delayed DDPG**](https://arxiv.org/abs/1802.09477) (**TD3**) are re-implemented in [**PyTorch**](https://arxiv.org/abs/1912.01703) on [**OpenAI Gym**](https://arxiv.org/abs/1606.01540)'s [**Atari**](https://arxiv.org/abs/1207.4708) (`PongNoFrameskip-v4`, `BoxingNoFrameskip-v4`, `BreakoutNoFrameskip-v4`, `VideoPinball-ramNoFrameskip-v4`) and [**MuJoCo**](https://ieeexplore.ieee.org/document/6386109) (`Hopper-v2`, `HalfCheetah-v2`, `Ant-v2`, `Humanoid-v2`), not strictly compared with [**OpenAI Baselines**](https://github.com/openai/baselines), [**Dopamine**](https://github.com/google/dopamine), [**Spinning Up**](https://github.com/openai/spinningup) and [**Tianshou**](https://github.com/thu-ml/tianshou).

![overview](assets/overview.png)

## Demos
||Pong|Boxing|Breakout|Pinball|
|:-:|:-:|:-:|:-:|:-:|
|DQN|![Pong_DQN](outputs/PongNoFrameskip-v4/DQN/video/Pong_DQN.gif)|![Boxing_DQN](outputs/BoxingNoFrameskip-v4/DQN/video/Boxing_DQN.gif)|![Breakout_DQN](outputs/BreakoutNoFrameskip-v4/DQN/video/Breakout_DQN.gif)|![Pinball_DQN](outputs/VideoPinball-ramNoFrameskip-v4/DQN/video/Pinball_DQN.gif)|
|D3QN|![Pong_D3QN](outputs/PongNoFrameskip-v4/D3QN/video/Pong_D3QN.gif)|![Boxing_D3QN](outputs/BoxingNoFrameskip-v4/D3QN/video/Boxing_D3QN.gif)|![Breakout_D3QN](outputs/BreakoutNoFrameskip-v4/D3QN/video/Breakout_D3QN.gif)|![Pinball_D3QN](outputs/VideoPinball-ramNoFrameskip-v4/D3QN/video/Pinball_D3QN.gif)|

||Hopper|HalfCheetah|Ant|Humanoid|
|:-:|:-:|:-:|:-:|:-:|
|DDPG|![Hopper_DDPG](outputs/Hopper-v2/DDPG/video/Hopper_DDPG.gif)|![HalfCheetah_DDPG](outputs/HalfCheetah-v2/DDPG/video/HalfCheetah_DDPG.gif)|![Ant_DDPG](outputs/Ant-v2/DDPG/video/Ant_DDPG.gif)|![Humanoid_DDPG](outputs/Humanoid-v2/DDPG/video/Humanoid_DDPG.gif)|
|TD3|![Hopper_TD3](outputs/Hopper-v2/TD3/video/Hopper_TD3.gif)|![HalfCheetah_TD3](outputs/HalfCheetah-v2/TD3/video/HalfCheetah_TD3.gif)|![Ant_TD3](outputs/Ant-v2/TD3/video/Ant_TD3.gif)|![Humanoid_TD3](outputs/Humanoid-v2/TD3/video/Humanoid_TD3.gif)|

## Dependencies
Main: python3.8, gym0.26.2, mujoco2.1.0.
```bash
# create conda environment
conda create -n rl python=3.8
conda activate rl
# install gym
pip install gym==0.26.2
# install gym[atari]
pip install gym[atari]
pip install gym[accept-rom-license]
# install gym[mujoco]
pip install gym[mujoco]
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
tar -zxvf mujoco210-linux-x86_64.tar.gz
mkdir ~/.mujoco
mv mujoco210 ~/.mujoco/mujoco210
rm mujoco210-linux-x86_64.tar.gz
pip install -U 'mujoco-py<2.2,>=2.1'
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
pip install Cython==3.0.0a10
# install other dependencies
pip install tqdm
pip install numpy
pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install tensorboard
pip install opencv-python
pip install einops
```

## Get Started
```bash
python run.py --env_name [env_name] [--improve] [--test] --seed [seed] --device [device] [--debug] [--gui] [--video]
```

## Results
> Full results refer to `outputs`. Note that the hyperparameters of the algorithms vary across different implementations. Also, the metric used is not strictly the same (e.g. average testing score with 10 trials is used in RLDemo, while Tianshou uses the max average validation score in the last training 1M/10M timesteps). In addition, other popular implementations have no individual D3QN implementation, so Rainbow instead is selected for the D3QN's performance. Besides, other popular implementations have no individual results for the RAM version of Pinball environment, so its normal image version is selected.

|DQN|Pong|Boxing|Breakout|Pinball|
|:-:|:-:|:-:|:-:|:-:|
|OpenAI Baselines|16.5|-|**131.5**|-|
|Dopamine|9.8|\~77|92.2|**\~65000**|
|Tianshou|20.2|-|**133.5**|-|
|RLDemo|**21.0**|**95.4**|9.6|6250.0|

|D3QN/Rainbow|Pong|Boxing|Breakout|Pinball|
|:-:|:-:|:-:|:-:|:-:|
|OpenAI Baselines|-|-|-|-|
|Dopamine|19.1|**~99**|47.9|**\~465000**|
|Tianshou|20.2|-|**440.4**|-|
|RLDemo|**21.0**|84.6|6.2|4376.6|

|DDPG|Hopper|HalfCheetah|Ant|Humanoid|
|:-:|:-:|:-:|:-:|:-:|
|Spinning Up|\~1800|**\~11000**|\~840|-|
|Tianshou|2197.0|**11718.7**|990.4|177.3|
|RLDemo|**3289.2**|8720.5|**2685.3**|**2401.4**|

|TD3|Hopper|HalfCheetah|Ant|Humanoid|
|:-:|:-:|:-:|:-:|:-:|
|Spinning Up|\~2860|\~9750|\~3800|-|
|Tianshou|**3472.2**|10201.2|**5116.4**|**5189.5**|
|RLDemo|1205.3|**12254.4**|**5058.1**|**5206.4**|

||Pong|Boxing|Breakout|Pinball|
|:-:|:-:|:-:|:-:|:-:|
|Loss|![Pong_loss](outputs/PongNoFrameskip-v4/tb/Pong_loss.png)|![Boxing_loss](outputs/BoxingNoFrameskip-v4/tb/Boxing_loss.png)|![Breakout_loss](outputs/BreakoutNoFrameskip-v4/tb/Breakout_loss.png)|![Pinball_loss](outputs/VideoPinball-ramNoFrameskip-v4/tb/Pinball_loss.png)|
|Validation Score|![Pong_score](outputs/PongNoFrameskip-v4/tb/Pong_score.png)|![Boxing_score](outputs/BoxingNoFrameskip-v4/tb/Boxing_score.png)|![Breakout_score](outputs/BreakoutNoFrameskip-v4/tb/Breakout_score.png)|![Pinball_score](outputs/VideoPinball-ramNoFrameskip-v4/tb/Pinball_score.png)|
|Validation Return|![Pong_return](outputs/PongNoFrameskip-v4/tb/Pong_return.png)|![Boxing_return](outputs/BoxingNoFrameskip-v4/tb/Boxing_return.png)|![Breakout_return](outputs/BreakoutNoFrameskip-v4/tb/Breakout_return.png)|![Pinball_return](outputs/VideoPinball-ramNoFrameskip-v4/tb/Pinball_return.png)|
|Validation Iterations|![Pong_iterations](outputs/PongNoFrameskip-v4/tb/Pong_iterations.png)|![Boxing_iterations](outputs/BoxingNoFrameskip-v4/tb/Boxing_iterations.png)|![Breakout_iterations](outputs/BreakoutNoFrameskip-v4/tb/Breakout_iterations.png)|![Pinball_iterations](outputs/VideoPinball-ramNoFrameskip-v4/tb/Pinball_iterations.png)|

||Hopper|HalfCheetah|Ant|Humanoid|
|:-:|:-:|:-:|:-:|:-:|
|Actor Loss|![Hopper_actor_loss](outputs/Hopper-v2/tb/Hopper_actor_loss.png)|![HalfCheetah_actor_loss](outputs/HalfCheetah-v2/tb/HalfCheetah_actor_loss.png)|![Ant_actor_loss](outputs/Ant-v2/tb/Ant_actor_loss.png)|![Humanoid_actor_loss](outputs/Humanoid-v2/tb/Humanoid_actor_loss.png)|
|Critic Loss|![Hopper_critic_loss](outputs/Hopper-v2/tb/Hopper_critic_loss.png)|![HalfCheetah_critic_loss](outputs/HalfCheetah-v2/tb/HalfCheetah_critic_loss.png)|![Ant_critic_loss](outputs/Ant-v2/tb/Ant_critic_loss.png)|![Humanoid_critic_loss](outputs/Humanoid-v2/tb/Humanoid_critic_loss.png)|
|Validation Score|![Hopper_score](outputs/Hopper-v2/tb/Hopper_score.png)|![HalfCheetah_score](outputs/HalfCheetah-v2/tb/HalfCheetah_score.png)|![Ant_score](outputs/Ant-v2/tb/Ant_score.png)|![Humanoid_score](outputs/Humanoid-v2/tb/Humanoid_score.png)|
|Validation Return|![Hopper_return](outputs/Hopper-v2/tb/Hopper_return.png)|![HalfCheetah_return](outputs/HalfCheetah-v2/tb/HalfCheetah_return.png)|![Ant_return](outputs/Ant-v2/tb/Ant_return.png)|![Humanoid_return](outputs/Humanoid-v2/tb/Humanoid_return.png)|
|Validation Iterations|![Hopper_iterations](outputs/Hopper-v2/tb/Hopper_iterations.png)|![HalfCheetah_iterations](outputs/HalfCheetah-v2/tb/HalfCheetah_iterations.png)|![Ant_iterations](outputs/Ant-v2/tb/Ant_iterations.png)|![Humanoid_iterations](outputs/Humanoid-v2/tb/Humanoid_iterations.png)|
