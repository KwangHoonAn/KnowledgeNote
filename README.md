# KnowledgeNote

This repository is to make a note for potential interview related to NLP or Machine Learning position.

## Why not sinusoidal function (cos, sin) are not widely used in neural network as an activation function?

Following is my assumption.
Unlike sigmoid function, cos and sin function are periodically oscilating between -1 ~ 1 


**In neural networks with sigmoid function, positive value fires activation function to output as a positive reponse to input data to target class.**

 This is because sigmoid function will have a converged value to 1 or 0 as input values increase.
 However, two sinusoidal functions may not recognize positive response as positive since both negative response and positive can be in the same numerical value outputs.

 I have implemented experiments (DIfferentActivations.ipynb) on MNIST data to observe loss and accuracy based on different activations.
 My experiment shows cos and sin functions somehow produces faster learning, which is contradict to my assumption.

 I still need to read more papers and literactures to grasp clear concepts on this..

## Probability
What is prior probability P(D)?

Prior probability is the probability that **we already know before we infer to get posterior probability**

For example, we have 5 documents, [D_sport, D_sport, D_medical, D_medical, D_engineering]
Given our information about the class, we now know that we have 2 sports, 2 medical, 1 engineering document.
We can simply estimate prior probability of each as

P(D_sport) = N_sport / Total Document Numbers, where N_sport = 2 and Total Document Number = 5

To be short, we can consider the prior probability as **belief of the answer or class without any query information**


What is posterior probability P(D|'unseen article')

Simply, posterior probability is the probability that we want to know about the unseen document to identify its class. For generative model, we apply Bayesian rule to estimate posterior probability (prior*likelihood)