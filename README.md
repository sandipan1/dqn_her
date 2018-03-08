# dqn_her

Implementation of Hindsight Experience Replay with DQN

To implement HER we created a file dqn_her.py which is a copy of baselines.deepq’s simple.py along with the following changes in the learn() function - 

1. Added a new parameter **num_optimisation_steps** which is the number of optimisation steps performed after every train_freq number of episodes.
2. Made the input size of neural network double to incorporate goals along with state by changing in function **make_obs_ph**.
3. Defined variables **mean_100ep_max_reward** and **saved_mean_reward_diff** for episode’s max reward and mean reward difference because in HER the goals and initial state change in each episode. So for determining if model is improving we see difference between max reward possible for episode and the reward agent actually got in that episode. 
4. Environment **goal** is concatenated with states and feed into function act(). It is also concatenated with states when storing in replay buffer.
5. Defined a temporary buffer **episode_buffer** which stores all the transitions of the episode. Then at the end of the episode these transitions are concatenated with **goal_prime** (in our case final state reached in the episode) and stored in replay buffer. 

We created a bit-flipper environment in openai gym format and used this algorithm on it. 
Check out our blog for more details https://deeprobotics.wordpress.com/
