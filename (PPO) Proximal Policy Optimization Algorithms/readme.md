# Proximal Policy Optimization Algorithms (PPO)
This folder contains the implementation of paper entitled "Proximal Policy Optimization Algorithms" ("https://arxiv.org/pdf/1707.06347.pdf").
This contribution of this paper is about using (Proximal Policy Optimization Algorithms (PPO)) in problems of HIGH-DIMENSIONAL CONTINUOUS CONTROL (problems that require continious control through all the episode).

# Implementation details 
The implementataion is very similar to the GENERALIZED ADVANTAGE ESTIMATION (GAE) implementataion ("https://github.com/Abdelhamid-bouzid/Policy-Gradient/tree/main/GENERALIZED%20ADVANTAGE%20ESTIMATION") the only differnce is in the loss function of the actor:
- instead of new_log(prob)*A, it is [new_log(prob)/old_log(prob)]*A, where [new_log(prob)/old_log(prob)] is clipped between [1-eps, 1+eps]
