
> This directory contains the final project for **ECE‑465: Cloud Computing**, focusing on addressing CPU-bound bottlenecks with Ray and leveraging containerization using Kubernetes for scalable distributed training.


### Objective  
We address CPU-bound bottlenecks in reinforcement-learning training by building a scalable workflow that uses Ray for distributed roll-outs and Kubernetes for containerized deployment. Our aim is to demonstrate near-linear speed-up across multiple cores while preserving final policy quality, using a Tic-Tac-Toe Deep Q-Network (DQN) as a reproducible benchmark.



### Approach  
We refactored a single-CPU Q-table baseline into a distributed DQN that launches one head process and $k$ Ray actors to collect experience in parallel. We package the entire pipeline—including Python 3.10, Gym 0.26, Ray ≥ 2.0, and PyTorch ≥ 1.12—into a Docker image, then validate identical execution on a local Minikube cluster and a 12-vCPU Paperspace C7 instance managed by the Ray Operator. We sweep $k\in\{1,2,4,8,12\}$, log wall-clock time, episodes per second, and gradient updates, and evaluate checkpoints against uniform-random and self-play opponents.


### Directory Structure

```
.
├── Dockerfile
├── README.md
├── deployment.yaml
├── documentation.md
├── main.py
├── notebooks
│   └── train.ipynb
├── requirements.txt
└── service.yaml
```

<br>

### Results
Distributed roll-outs cut the time to train 100 k episodes from 295.8 s ($k=1$) to 228.6 s ($k=12$), yielding a 1.29× speed-up and throughput growth from 338 eps/s to 438 eps/s. Speed-up remains near-linear through $k=8$ (1.29×) before flattening, indicating diminishing returns due to CPU contention. Policy performance at 100 k episodes remained consistent across $k$: the agent won 60–65 % of games when moving first and 31–35 % when moving second, with mirrored self-play confirming a strong first-move bias. These findings validate that parallelism accelerates convergence without degrading final strategy quality.

<br>

### References


<br>
