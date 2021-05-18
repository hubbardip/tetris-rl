from nes_py.wrappers import JoypadSpace
import gym_tetris
from gym_tetris.actions import SIMPLE_MOVEMENT

env = gym_tetris.make('TetrisA-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

done = False
obs = env.reset()
while not done:
    obs, r, done, _ = env.step(env.action_space.sample())
    env.render()

env.close()
