
> This directory contains the final project for **ECE‑465: Cloud Computing**, focusing on addressing CPU-bound bottlenecks with Ray and leveraging containerization using Kubernetes for scalable distributed training.


### Objective  
Our goal is to overcome CPU-bound limitations in distributed training by harnessing Ray for efficient hyperparameter tuning and containerizing our workflow with Kubernetes. This project aims to build a scalable framework that can transition from simpler tasks, like tuning on MNIST, to more complex challenges such as reinforcement learning.



### Approach  
We begin by integrating Ray Tune to execute parallel hyperparameter searches on the MNIST dataset across multiple CPU cores. Once we establish a robust local workflow, we plan to containerize our process with Kubernetes, enabling deployment on cloud infrastructure with significantly more computing power. This setup will not only streamline the tuning process but also set the stage for scaling to CPU-intensive applications, including reinforcement learning tasks.


### Progress

- [x] **Parallel Hyperparameter Tuning with Ray:**  
  Set up and run distributed hyperparameter tuning on the MNIST dataset using Ray Tune.

- [x] **Containerization:**  
  Containerized the entire workflow and deployed it on a Minikube Kubernetes cluster.

- [ ] **Scaling on Cloud Infrastructure:**  
  Plan to extend containerization to a full cloud-based Kubernetes deployment using RayKube.

- [ ] **Transition to Reinforcement Learning:**  
  Aim to apply the established framework to CPU-intensive reinforcement learning tasks.


### Repository Structure

```
```

