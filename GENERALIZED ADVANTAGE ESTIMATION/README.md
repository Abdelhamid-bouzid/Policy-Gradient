# GENERALIZED ADVANTAGE ESTIMATION (GAE)
This folder contains the implementation of paper entitled "HIGH-DIMENSIONAL CONTINUOUS CONTROL USING GENERALIZED ADVANTAGE ESTIMATION" ("https://arxiv.org/pdf/1506.02438.pdf").
This contribution of this paper is about using (GENERALIZED ADVANTAGE ESTIMATION (GAE)) in problems of HIGH-DIMENSIONAL CONTINUOUS CONTROL (problems that require continious control through all the episode).

# The implementation pipeline:
The following figure displays the pipeline of the implementation of the GAE framework.

- The idea is to play one episode (continious control). During playing the agent saves all the transitions and values of of states (old values and actions probs).
- Then in a backward manner, the agnet computes the GAE for every state.
- Divide the memory to mini-batche, and for every batch:
  - recompute the new log probs 
  - recompute the new state values 
  - compute the actor loss 
  - compute the critic loss 
  - update.

![Image](Pipeline.JPG)
