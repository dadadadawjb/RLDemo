## Dependencies
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
