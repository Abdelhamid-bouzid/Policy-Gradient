# A2C Actor Critic with TD error
This folder contains the implementation of the papaer entitled "SAMPLE EFFICIENT ACTOR-CRITIC WITH EXPERIENCE REPLAY" ("https://arxiv.org/pdf/1611.01224.pdf"). The critic is the TD error of Q-learning between s and s'.
The performance of the framework proposed by this paper is not that good. the reason is there is no target critic that provides the target for the critic. Therefore, in order to improve this I suggest using a target critic like in deep Q-learning (which is actually proposed in new published papers such as actor soft critic).


# Implemenatation details:
- We used replay buffer.
- The actor acts based on its probabailties estimatation.
- the critic gives its crictic based on the TD error.

