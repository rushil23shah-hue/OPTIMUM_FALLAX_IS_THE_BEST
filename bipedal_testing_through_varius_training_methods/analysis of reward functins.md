# Reward Shaping in BipedalWalker

In this project I experimented with different reward functions in BipedalWalker to see how reward shaping changes the behaviour of the agent. 
The main aim was not only to make the agent cross the terrain, but to make it walk in a more stable and natural way. I tried multiple rewards one by one and observed how the movement changed.

## 1. Speed Reward

First I gave extra reward for forward speed. The idea was simple: if the walker moves faster in the forward direction, it should get more reward. When the speed factor was too high, the agent tried to move very fast and ended up falling.
After decreasing the factor, the agent was able to cross, but the movement was still mostly dragging and not proper walking. It was also slow, so I added a timestep penalty to encourage faster crossing.

After adding timestep penalty, the agent crossed faster, but the second leg was still dragging. From the graph, we can see that as training progressed, the speed increased and then became more constant,
which helped in getting more stability.

## 2. X and Y Coordinate Reward

After speed reward, I used X and Y coordinate based reward. For X coordinate, I did not reward the actual X position directly.
Instead, I rewarded the difference in X position in one timestep, so the agent got reward for actual forward progress.

For Y coordinate, I added a penalty for height error. This was because while dragging, jumping, or keeping one leg in air,
the torso height was changing a lot. So I penalized the difference in Y coordinate from the starting height. From the graph,
we can see that as training continued, the height error decreased. This gave a much more stable movement compared to the speed reward, but it was still somewhere between dragging and walking.

## 3. Angle and Angular Velocity Reward

Next I added angular values into the reward function. I noticed that while jumping or keeping one leg in air, the torso angle was changing a lot and the walker was also gaining angular velocity.
So I added penalties for body angle and angular velocity.

The idea was to make the torso stable and avoid jumping-based movement. After trial and error, I found a good set of factors. The final values were:
angle_factor = 3
angular_velocity_factor = 0.3 
After training with these values, the result was exceptional. The movement became proper walking and much smoother than previous experiments. 
Because of this, I treated this reward as my ideal stable walking reward.

## 4. Foot Contact Reward

After getting stable walking, I tried foot contact reward. This was one of the most interesting reward functions because it focused on how the feet touch the ground.

The reward function was:

proxy_reward = reward
             + single foot contact reward
             + alternate foot contact reward
             - no foot contact penalty
             - both feet contact penalty

Single foot contact helped the agent keep one foot on the ground at a time. Alternate foot contact encouraged left-right leg movement. No foot contact was penalized because it means the walker is flying or jumping.
Both feet contact was also penalized because it can lead to dragging.
Initially, when the factors were not balanced properly, the bot was doing jumping-like alternate leg movement but was not moving forward properly.
After experimenting with the factors, I found a better set and the movement became very good. It started showing alternate leg movement and moved much faster compared to the angular velocity model. 
It almost looked like slow running.
From the graph, we can see that alternate movement increased during training. Also, one-foot contact had the highest ratio compared to no-foot contact or both-feet contact,
which shows that the agent learned a better walking-like pattern.

## 5. Center of Mass Jitter Reward
The final reward I tried was based on jitteriness of center of mass. The idea was to reduce unnecessary shaking or bouncing of the walker.
To calculate center of mass, I used the hull and all leg parts of the walker. I calculated it using weighted average of their positions based on their mass, so heavier body parts affect the center of mass more.
The jitter reward mainly checked sudden movement in center of mass. I used vertical jitter and speed jitter:
com_jitter = vertical_jitter + 0.5 * speed_jitter
Then I added a penalty for this jitter. Initially, with heavy factors, the walker almost stopped moving and behaved like a statue until time ran out. After tuning the factor,
I found a good value and got smooth stable walking. It crossed properly, but the movement was still not as smooth as the ideal reward made from X, Y, angle and angular velocity.
From the graph, we can see that as episodes passed, the jitter decreased and the movement became more stable.

## Final Observation
Overall, each reward changed the behaviour of the agent in a different way. Speed reward helped it move faster but caused falling or dragging when not balanced. X and Y reward improved forward progress and height stability.
Angle and angular velocity reward gave the best natural walking movement. Foot contact reward improved leg alternation and gave faster movement. Center of mass jitter reward helped reduce shaking and made the movement smoother.
The main learning from this project was that reward shaping is very powerful, but small changes in reward factors can completely change the agent's behaviour. Sometimes the agent learns exactly what we want,
but sometimes it finds shortcuts like dragging, jumping, or standing still. So reward functions must be designed and tested carefully.
