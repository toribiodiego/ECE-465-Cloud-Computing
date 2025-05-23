
> This repository contains the code for **ECE 465: Cloud Computing**, a 3-credit graduate-level course at The Cooper Union for the Advancement of Science and Art, providing hands-on experience in cloud computing.

## Cloud-Computing  
**Independent Study, Spring 2025**  
**Instructor:** Prof. Rob Marano  
**Syllabus:** [View](https://robmarano.github.io/courses/ece465/2025/ece465-ind-study-syllabus-spring-2025.html)

### Overview

This independent study provides a practical introduction to cloud computing, focusing on distributed systems, containerization, and modern DevOps practices. Through in-depth exploration of topics like distributed training, multi-threaded programming, and cloud deployment techniques, students gain both theoretical insights and hands-on experience needed to design and scale production-ready applications.

### Material

The primary resource for the course is *Distributed Systems, 4th ed.* by Andrew Tanenbaum and Maarten van Steen, which lays the foundation for understanding the evolution of cloud computing from distributed systems. Complementing this, students will explore various online resources and hands-on examples that illuminate key technologies and best practices in modern cloud computing.

- Multi-processing and network programming techniques  
- Containerization with Docker and orchestration with Kubernetes  
- CI/CD pipelines for efficient development and deployment  
- Distributed architectures focusing on consistency, replication, and fault tolerance  
- Cloud-based deployment on virtual nodes and Kubernetes clusters

### Repository Structure  

```
.
├── Final_Project
│   ├── Dockerfile
│   ├── README.md
│   ├── deployment.yaml
│   ├── main.py
│   ├── notebooks
│   │   └── train.ipynb
│   ├── requirements.txt
│   └── service.yaml
├── PS01.ipynb
└── README.md
```

- **PS01.ipynb**: Dining Philosophers Problem (Concurrency and Deadlock Prevention)  

### Final Project

This project, *Distributed Reinforcement Learning for Tic-Tac-Toe*, explores how to scale CPU-bound reinforcement learning tasks using Ray, Kubernetes, and Docker. Inspired by production challenges in modern cloud environments, the goal is to transition from a single-CPU pipeline to a multi-core, multi-node setup that accelerates self-play simulations and policy updates. By leveraging containerization and distributed computing, the project aims to gain practical insights into resource optimization and scalable cloud deployment for machine learning applications.