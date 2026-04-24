---
title: "Project 1: Learn to Optimise"
course: "ELEN90088 — System Optimisation and Machine Learning"
parent: "[[SOML_Project_26S1]]"
---

# Project 1: Learn to Optimise

## Introduction

Learning to optimize is an emerging approach that leverages machine learning to develop optimization methods, aiming at reducing the laborious iterations of hand engineering. It automates the design of an optimization method based on its performance on a set of training problems. This data-driven procedure generates methods that can efficiently solve problems similar to those in training. In sharp contrast, the typical and traditional designs of optimization methods are theory-driven, so they obtain performance guarantees over the classes of problems specified by the theory. This new paradigm has motivated a community of researchers to explore the potential of the learning-to-optimize method in various fields, such as wireless communication, image restoration and reconstruction, medical and biological imaging, etc.

One of the learning-to-optimize schemes is known as *deep unrolling/unfolding*. Consider a general iterative optimization algorithm that targets regressing an unknown signal $x^*$ from the observation $d$ with the updating form

$$
x^{k+1} = T(x^k; d), \quad k = 0, 1, 2, \ldots \tag{1}
$$

where the optimization algorithm $T(\cdot)$ takes $d$ as the input and iterates over $x^k$. Deep unrolling unrolls and truncates an optimization algorithm, which can be viewed as a network, shown in Figure 1.

The new updates take the form

$$
x^{k+1} = T_{\theta^k}(x^k; d), \quad k = 0, 1, 2, \ldots, K-1, \tag{2}
$$

where $K$ is the depth of the unrolled network, and $\theta^k$ are the learnable weights in the $k$-th layer. An unrolled network example is illustrated in Figure 1. Upon establishing this form, the parameters are learned by an end-to-end approach:

$$
\min_{\{\theta^k\}_{k=0}^{K-1}} \mathcal{L}\left( x^K\left(\{\theta^k\}_{k=0}^{K-1}\right); d \right), \tag{3}
$$

where $\mathcal{L}$ is the loss function we use in training. We emphasize the distinction that the parameters in unrolled schemes are trained end-to-end using the iterate $x^K$ as a function of each $\theta^k$. We also point out that although the structure in Figure 1 resembles that of a feedforward deep neural network, the fine structure at each "layer" is *not* generally the same as the structure of a neuron (i.e. linear functions followed by an activation function).

In the following, we will introduce how to implement the deep unfolding for a specific optimization problem, inspired by a detection problem in wireless communication scenarios.

![[Figure_1.png]]

**Figure 1:** A common approach in various learning-to-optimize schemes is to form feed-forward networks by unrolling an iterative algorithm, truncated to $K$ iterations, and tuning parameters at each layer/iteration $k$. This generalizes the updating formula $x^{k+1} = T(x^k; d)$ to include a dependence on weights $\theta^k$, denoted by a subscript.

## Deep-unfolded PG (DU-PG) for integer programming

Let us consider the following problem

$$
\begin{aligned}
\underset{x}{\text{minimize}} \quad & \tfrac{1}{2}\|Ax - y\|_2^2, \\
\text{subject to} \quad & x \in \{-1, +1\}^n,
\end{aligned} \tag{4}
$$

where $A \in \mathbb{R}^{n \times n}$ is a given matrix and $\|\cdot\|_2$ represents the Euclidean norm. We assume that $y$ is stochastically generated as $y = Ax + z$, where $x$ is a vector sampled from $\{-1, +1\}^n$ uniformly at random and $z \in \mathbb{R}^n$ consists of i.i.d. Gaussian random variables following $\mathcal{N}(0, \sigma^2)$.

The problem has a quadratic cost, but the optimization variables are constrained to a finite set. This makes the problem non-convex and finding the global minimum is NP-hard, which is computationally intractable for large-scale parameters. In general, we need to solve the problem approximately. We here exploit a variant of the projected gradient descent (PG) algorithm to solve (4) approximately. The PG-like algorithm can be described by the following recursive formula:

$$
\begin{align}
r^k &= s^k + \gamma A^T (y - A s^k), \tag{5}
\\ \\
s^{k+1} &= \tanh\left( \alpha r^k \right), \tag{6}
\end{align}
$$

where $k = 0, 1, \ldots, K-1$ is the iteration index and $\tanh(\cdot)$ is calculated element-wisely. The PG algorithm consists of two computational steps for each iteration. In the gradient descent step (5), a search point moves in the opposite direction to the gradient of the objective function. The parameter $\gamma$ controls the step size causing a critical influence on the convergence behavior. In the projection step (6), soft-projection based on the hyperbolic tangent function is applied to the search point to obtain a new search point nearly rounded to binary values, where $\alpha$ controls the softness of the soft projection. Specifically, the projection step is not the projection to the binary symbols $\{-1, +1\}$. This is because the true projection to discrete values results in insufficient convergence behavior in a minimization process.

According to the data-driven learning-to-optimize framework, we can embed trainable parameters into the variant PG algorithm. The resultant deep-unfolded PG (DU-PG) algorithm can be described by the following recursions:

$$
\begin{align}
r^k &= s^k + \gamma^k A^T (y - A s^k), \tag{7}
\\ \\
s^{k+1} &= \tanh\left( \alpha r^k \right), \tag{8}
\end{align}
$$

where $\{\gamma^k\}_{k=0}^{K-1}$ are trainable parameters, which play key roles in the gradient descent step by adjusting its step size adaptively. Besides, the parameter $\alpha$ is treated as a pre-determined hyperparameter. Finally, the DU-PG algorithm with $K$ iterations outputs an estimate of $x$, which is denoted as $\hat{x}(y) = s^K$.

In the data generation phase, $N$ training data pairs $\{(x_i, y_i)\}_{i=1}^{N}$ are i.i.d. generated at random. More specifically, $x_i \in \{-1, +1\}^n$ is generated uniformly at random and the corresponding $y_i$ is then generated according to $y_i = A x_i + z_i$ with a given matrix $A$. Note that $A$ is randomly generated before the dataset generation procedure, and each element of $A$ is sampled from a Gaussian distribution $\mathcal{N}(0, 1)$.

For the training phase, the whole data set is equally divided into $N/D$ mini-batches, each of which is composed of $D$ training data pairs $\mathcal{D} := \{(x_1, y_1), (x_2, y_2), \ldots, (x_D, y_D)\}$. A mini-batch is fed to the DU-PG algorithm to calculate the squared loss function which is given by

$$
\mathcal{L}(\theta^{K-1}) = \frac{1}{D} \sum_{i=1}^{D} \| x_i - \hat{x}(y_i) \|_2^2, \tag{9}
$$

where $\theta^t := \{\gamma^0, \gamma^1, \ldots, \gamma^k\}$ is the collection of trainable parameters up to the $k$-th layer. Then, an SGD-type algorithm is employed to minimize (9) by updating $\theta$. It is also worth noting that two training methods are considered as follows:

- **Single-shot training:** Let $k = K - 1$ and all of the parameters are trained simultaneously.
- **Incremental training:** The parameters are sequentially trained from $\theta^0$ to $\theta^{K-1}$ in an incremental manner. At first, $\theta^0$ is trained by minimizing $\mathcal{L}(\theta^0)$. After finishing the training of $\theta^0$, the values of trainable parameters in $\theta^0$ are copied to the corresponding parameters in $\theta^0$. In other words, the results of the training for $\theta^0$ are taken over to $\theta^1$ as the initial values. All of the mini-batches will be processed during each round of the incremental training procedure.

Finally, the recommended configuration of the parameters used in the experiment is shown below. The noise variance is fixed to $\sigma^2 = 4$. The number of iterations (layer) of the DU-PG algorithm is $T = 20$. In the training process, there are 100 mini-batches each with size $D = 200$. You can try the Adam optimizer with a learning rate of $5 \times 10^{-4}$ for the parameter updating. The trainable parameters are initialized as $\gamma_k = 10^{-4}$ for $k = 0, 1, \ldots, K - 1$ and the softness parameter is fixed to $\alpha = 8.5$.

Complete the tasks in the following two parts. In Part 1, you will implement the deep unrolling for the integer programming problem, following the detailed instructions in the above section. In Part 2, you will apply the idea of learning-to-optimize to a new algorithm.

## Part 1

### Implementation of PG

Implement the PG algorithm introduced in (5), (6). Plot the mean square error (MSE) between $x_i$ and $\hat{x}(y_i)$ averaged on the training and test dataset, respectively. If the estimation is $\hat{x}(y) = \operatorname{sgn}(s^K)$, where $\operatorname{sgn}(\cdot)$ is the sign function, plot the prediction accuracy for the test dataset.

### Implementation of DU-PG

Implement the improved DU-PG algorithm with the detailed instruction given in the above section. Use both the single-shot and incremental training methods. Provide performance comparisons between DU-PG algorithm (using both training methods) with the original variant PG algorithm. Discuss and explain your results, with an emphasis on how this learning-to-optimize scheme influences the original optimization algorithm.

### (Optional with bonus points) MIMO Detector Using DU-PG

Consider a practical Multiple Input Multiple Output (MIMO) communication system with additive white Gaussian noise (AWGN), which can be simplified as

$$
y = H x + z, \tag{10}
$$

where $H \in \mathbb{R}^{m \times n}$ is the channel matrix, $x \in \{-1, +1\}^n$ is the transmitted message vector, $z \in \mathbb{R}^m$ is the channel noise and each element of which follows $\mathcal{N}(0, \sigma^2)$, and $y \in \mathbb{R}^m$ is the received signal vector. To recover $x$ given $H$ and $y$, we exploit the variant PG algorithm as the MIMO detector:

$$
\begin{align}
r^k &= s^k + \gamma W (y - H s^k), \tag{11}
\\ \\
s^{k+1} &= \tanh\left( \alpha r^k \right), \tag{12}
\end{align}
$$

where $W$ is the Moore–Penrose pseudo inverse matrix of $H$, i.e.,

$$
W := H^T (H H^T)^{-1}.
$$

Similarly, the detector based on the DU-PG algorithm can be described as

$$
\begin{align}
r^k &= s^k + \gamma^k W (y - H s^k), \tag{13}
\\ \\
s^{k+1} &= \tanh\left( |\alpha^k| r^k \right), \tag{14}
\end{align}
$$

where $\{\gamma^k\}_{k=0}^{K-1}$ and $\{\alpha^k\}_{k=0}^{K-1}$ are trainable parameters. Again, implement the DU-PG-detector and analyze your results with the same requirements presented in the last two subsections.

## Part II

### Soft Nearest Neighbor

Consider the following Soft Nearest Neighbor (SNN) algorithm for regression. Let $\{(x_i, y_i)\}_{i=1}^{n}$ be the training dataset, where the feature vector $x_i \in \mathbb{R}^d$ and the label $y_i \in \mathbb{R}$ for all $i = 1, \ldots, n$. The prediction $\hat{y}$ for a test feature vector $x$ is given by

$$
\hat{y} = \sum_{i=1}^{n} \omega_i y_i, \tag{15}
$$

where $\omega_i$ is defined as

$$
\omega_i = \frac{e^{-\|x - x_i\|_2^2}}{\sum_{i=1}^{n} e^{-\|x - x_i\|_2^2}}. \tag{16}
$$

Now, try to incorporate the learning-to-optimize idea into the SNN algorithm. To be more specific, in (16) you can add one or several trainable parameters, after training which an improved SNN algorithm can be obtained. For the implementation where we offer a high degree of freedom, you are expected to get inspiration from Part I which can help design your classification problem, dataset, loss function, setting of hyperparameters, training method, etc. Illustrate your implementation details, and further present and discuss the results you obtain.

## Useful Resources

1. T. Chen, X. Chen, W. Chen, Z. Wang, H. Heaton, J. Liu, and W. Yin, "Learning to optimize: A primer and a benchmark," *The Journal of Machine Learning Research*, vol. 23, no. 1, pp. 8562–8620, 2022.
2. S. Takabe, M. Imanishi, T. Wadayama, and K. Hayashi, "Deep learning-aided projected gradient detector for massive overloaded MIMO channels," in *IEEE International Conference on Communications (ICC)*, 2019, pp. 1–6.
