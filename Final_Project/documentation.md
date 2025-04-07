> This document serves as a progress report for our final project, which explores how to scale CPU-bound training tasks using modern cloud technologies. 

### Project Overview

Our project, *Distributed Reinforcement Learning for Tic-Tac-Toe*, transforms a basic single-CPU training process into a scalable system capable of distributed training across multiple cores and nodes. By integrating Docker for containerization, Kubernetes for orchestration, and Ray for distributed computing, we can efficiently manage and scale our reinforcement learning tasks.

#### Key Components

- **Ray:** Used to manage distributed hyperparameter tuning and training.
- **Docker:** Packages our application into a portable container.
- **Kubernetes:** Orchestrates our containers, with current testing on Minikube.


### Containerization

We containerized our training process to ensure consistent behavior across all environments. Our `Dockerfile` serves as the blueprint for this, defining everything needed to run our application reliably. Specifically, it achieves the following:

- **Base Image:** It starts with the official Python 3.9 slim image.
- **Dependencies:** The file installs essential system packages (such as GCC) and all Python dependencies listed in our `requirements.txt`.
- **Application Code:** All source code is copied into the container, ensuring that our application is self-contained.
- **Execution:** The container is set to execute our main script (`main.py`), which kicks off the Ray distributed training process.
- **Port Exposure:** Port `8265` is exposed for the Ray dashboard, allowing easy access to monitoring tools.

This containerized setup not only guarantees that our application runs the same way everywhere—from local testing to cloud deployments—but also simplifies scaling on systems with large numbers of CPU cores.

### Ray

We use [Ray](https://ray.io/) to seamlessly scale our distributed training processes. In our project, Ray allows us to:

- **Initialization:** Our `main.py` script begins by initializing Ray, ensuring any existing sessions are properly closed before starting a new one.
- **Task Distribution:** Ray’s Tune module distributes hyperparameter tuning tasks across available CPUs, allowing us to explore different configurations simultaneously.
- **Resource Management:** The script allocates resources—such as CPUs and GPUs—to each task, ensuring efficient parallel execution.
- **Logging and Monitoring:** We log training metrics using WandB, and Ray’s dashboard provides a clear view of system performance.



### Kubernetes & Minikube

We use Kubernetes to deploy and manage our containerized application, and we're currently testing our setup on Minikube—a lightweight, local Kubernetes environment. Our setup is as follows:

- **Deployment Definition:**  
  Our `deployment.yaml` file outlines the desired state of our application. It specifies:
  - The number of replicas (currently 1)
  - The container image (`mnist-ray-tune:latest`)
  - Resource requests and limits to ensure smooth operation
  - Volume mounts to load environment variables

- **Service Exposure:**  
  The `service.yaml` file creates a Kubernetes Service that makes our application accessible outside the cluster. It uses the NodePort service type to forward traffic from port `30000` on the host to port `8265` in the container.

- **Minikube:**  
  We run our Kubernetes cluster on Minikube to validate our deployment process in a controlled, local environment before scaling to larger, production-like setups.

### Progress Checklist

**Achievements:**
- [x] Containerized the entire application using Docker.
- [x] Integrated Ray for managing distributed tasks and hyperparameter tuning.
- [x] Deployed the containerized application on a local Kubernetes cluster using Minikube, validating our cloud-like setup.

**Next Steps:**
- [ ] Test multi-replica deployments on a larger Kubernetes cluster.
- [ ] Enhance monitoring by integrating additional tools for tracking resource usage and performance.
- [ ] Optimize hyperparameter tuning and explore more Ray features for scaling training.