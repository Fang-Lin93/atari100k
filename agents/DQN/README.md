# COMP5212 project (May/2022) - Learn to Play Atari Games with Data-Efficient Reinforcement learning



Here I use Q-learning with multi-step value target and 
[distributed prioritized experienced replay](https://arxiv.org/pdf/1803.00933.pdf)
to train a sample-efficient agent. It works for a single laptop without GPUs. The support of GPU 
usage will be updated later.

I mainly refer to [EfficientZero](https://github.com/YeWR/EfficientZero) for the training environments and 
representation networks. But I totally remove the MCTS parts for simplicity. And I add a QueueWorker to
focus on batch data preparing.

Deprecation warning due to gym seeds and video recorders may be existed.

Make sure the working directory is under **atari100k** and run

`
Python train.py
`

The results should be saved in a "results" folder under **atari100k**. Test process will always save videos.
Turn it off by tune parameters in train.py (Hyper-parameters are far from optimal and need to be tuned)
