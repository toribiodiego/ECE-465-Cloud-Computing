This document shows **three** self-contained ways to reproduce the distributed
Tic-Tac-Toe DQN experiments described in the report:

1. **One-box Docker run** – quickest path; no Kubernetes required.  
2. **Minikube smoke test** – validates the Ray Operator manifest on a laptop.  
3. **Paperspace C7 cloud run** – replicates the full 12-vCPU sweep used in the
   paper, including artefact export.

Each path uses the *same* Docker image, so results are bit-for-bit identical
aside from wall-clock timing.

---

## 0 . Prerequisites

| Tool | Version                               | Notes                                   |
|------|---------------------------------------|-----------------------------------------|
| Docker | ≥ 24.0 CE                           | Make sure your user is in the `docker` group. |
| Git    | ≥ 2.34                              |                                           |
| Minikube & kubectl | *(Minikube path only)*  | Tested with Minikube v1.32, Kubernetes v1.29 |
| Paperspace account | *(Cloud path only)*     | One C7 VM (12 vCPU, Ubuntu 22.04)        |

Clone the repository once:

```bash
git clone https://github.com/toribiodiego/ECE-465-Cloud-Computing.git
cd ECE-465-Cloud-Computing/Final_Project/dqn
````

---

## 1 . One-Box Docker Run

The container encapsulates Python 3.10, Ray 2.46, PyTorch 2.7, Gym 0.26,
Matplotlib, and Seaborn.

```bash
# build (≈ 700 MB, 2–3 min on broadband)
docker build -t dqn-tictactoe .

# run an 8-actor job; artefacts saved to ./outputs on the host
mkdir -p $(pwd)/outputs
docker run --rm -it \
  --cpus 12 \
  -v $(pwd)/outputs:/workspace \
  dqn-tictactoe \
  bash -c "./run.sh --actors 8"
```

*Expected runtime*: **≈ 4 min** for 100 k episodes (8 actors) on a modern
12-core CPU.

All artefacts—`plots/`, `checkpoints_k8/`, and `sweep.txt`—appear in
`./outputs`.

---

## 2 . Minikube Smoke Test (4 CPU Laptop)

This path confirms that the Kubernetes manifests work before running in the
cloud.

```bash
# start a 4-core local cluster
minikube start --cpus 4 --memory 6g

# install Ray Operator CRD and controller
kubectl apply -f k8s/ray-operator.yaml

# deploy one head + four worker pods
kubectl apply -f k8s/local-ray-cluster.yaml
```

Monitor training:

```bash
HEAD=$(kubectl get pods -l ray.io/cluster=head -o jsonpath='{.items[0].metadata.name}')
kubectl logs -f $HEAD
```

Export artefacts from the head pod when finished:

```bash
kubectl cp $HEAD:/workspace ./minikube-outputs
```

---

## 3 . Paperspace C7 Cloud Run

### 3.1 Create the VM

| Option       | Setting                     |
| ------------ | --------------------------- |
| Machine type | **C7** (12 vCPU, 32 GB RAM) |
| OS template  | Ubuntu 22.04                |
| Disk size    | 100 GB (default OK)         |
| Public IP    | **Enabled** (SSH access)    |
| SSH key      | paste your `id_ed25519.pub` |

### 3.2 Bootstrap the environment

```bash
ssh paperspace@<VM-IP>   # use the key you added

# install Docker & Git
sudo apt update && sudo apt install -y docker.io git
sudo usermod -aG docker $USER && newgrp docker

# clone and build
git clone https://github.com/toribiodiego/ECE-465-Cloud-Computing.git
cd ECE-465-Cloud-Computing/Final_Project/dqn
docker build -t dqn-tictactoe .
```

### 3.3 Run the full sweep

```bash
./run.sh                 # k = 1, 2, 4, 8, 12
zip -r phase3_outputs.zip checkpoints_* plots sweep.txt
```

### 3.4 Download artefacts

On **your laptop**:

```bash
scp -i ~/.ssh/id_ed25519 \
    paperspace@<VM-IP>:~/ECE-465-Cloud-Computing/Final_Project/dqn/phase3_outputs.zip .
unzip phase3_outputs.zip
```