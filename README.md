# RAdam

Detailed readme file is still in process. 

In this paper, we study the problem why we need warmup for Adam and identifies the adaptive learning rate has an undesirably large variance in the early stage. 
We verifies our hypothesis both theoretical and empirically, and propose RAdam, which achieves comparable performance with Adam+Warmup. 
Since our intuition and design is orthogonal to many other techniques, it can be integrated with them.
For example, since the warmup is originally to handle the variance of gradient, RAdam may lead to better performance by applying both linear warmup and rectification in some applications. 

We are in an early-release beta. Expect some adventures and rough edges.

## a simple guidance:

1. Directly replace Adam with RAdam first without changing any settings (if Adam works with some setting, it's likely RAdam also works with that). it is worth mentioning that, if you are using Adam with warmup, try RAdam with warmup first (instead of RAdam without warmup). 
2. Further tune hyper-parameters for a better performance. 
