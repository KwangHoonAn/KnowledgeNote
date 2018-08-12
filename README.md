# KnowledgeNote

This repository is to make a note for potential interview related to NLP or Machine Learning position.

## Probability
What is prior probability P(D)?

Prior probability is the probability that **we already know before we infer to get posterior probability**

For example, we have 5 documents, [D_sport, D_sport, D_medical, D_medical, D_engineering]
Given our information about the class, we now know that we have 2 sports, 2 medical, 1 engineering document.
We can simply estimate prior probability of each as

![first equation] (https://latex.codecogs.com/gif.latex?%5Cfrac%7BN_%7Bsport%7D%7D%7BTotal%20Document%20Num%7D)

To be short, we can consider the prior probability as **belief of the answer without any query information**

