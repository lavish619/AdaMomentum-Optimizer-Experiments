# AdaMomentum Optimizer Experiments

This repository cross-verifies the results of AdaMomentum optimization algorithm on Image Classification Task on CIFAR10 and CIFAR100 Datasets for the Resnet34 and Densenet121 Architecture as given in the paper. 

> **Adapting Stepsizes by Momentumized Gradients Improves Optimization and Generalization** <br> Yizhou Wangy, Yue Kangz, Can Qiny, Yi Xuy,
Huan Wangy, Yulun Zhangy, and Yun Fuy <br> <br> **Abstract**  
Adaptive gradient methods, such as Adam, have achieved tremendous success in machine learning. Scaling gradients by square roots of the running averages of squared past gradients, such methods are able to attain rapid training of modern deep neural networks. Nevertheless, they are observed to generalize worse than stochastic gradient descent (SGD) and tend to be trapped in local minima at an early stage during training. Intriguingly, we discover that substituting the gradient in the preconditioner term with the momentumized version in Adam can well solve the issues. The intuition is that gradient with momentum contains more accurate directional information and therefore its second moment estimation is a better choice for scaling than raw gradient’s. Thereby we propose AdaMomentum as a new optimizer reaching the goal of training faster while generalizing better. We further develop a theory to back up the improvement in optimization and generalization and provide convergence guarantee under both convex and nonconvex settings. Extensive experiments on various models and tasks demonstrate that AdaMomentum exhibits comparable performance to SGD on vision tasks, and achieves state-of-the-art results consistently on other tasks including language processing.

[Paper](https://arxiv.org/abs/2106.11514)  

## References
[1] Wangy Y., Kangz Y., Qiny C., Xuy Y., Wangy H., Zhangy Y. and Fuy Y. (2021). Adapting Stepsizes by Momentumized Gradients Improves Optimization and Generalization. _arXiv preprint, arXiv:2106.11514._

