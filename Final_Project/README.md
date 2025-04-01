


## Project Overview and Progress

**Current Focus:**  
We're currently exploring distributed hyperparameter tuning using Ray Tune on the MNIST dataset. This allows us to familiarize ourselves with the Ray framework, especially for CPU-bound tasks, by parallelizing training trials across all available cores.

**What We've Accomplished So Far:**

- **Ray Tune Integration:**  
  We've integrated Ray Tune to run 12 parallel experiments across our 12-core machine, efficiently distributing the hyperparameter search space.

- **Logging Improvements:**  
  Metrics are being logged to Weights & Biases (WandB), and we've implemented configuration changes (including environment variable settings and monkey patching) to suppress excessive console output from WandB.

- **Results Presentation:**  
  After completing the tuning, results are summarized and printed as a DataFrame table, giving us clear insights into the performance of each trial.

**Scaffolding Towards Our End Goal:**

- **Containerization & Cloud Scaling:**  
  With our current setup validated locally on MNIST, our next step is to containerize this process (using tools like RayKube) so we can leverage cloud infrastructure with many more cores. This will help us further accelerate CPU-bound processes.

- **Transition to Reinforcement Learning (RL):**  
  Our long-term goal is to tackle CPU-intensive experience rollouts in RL. Once we move from MNIST to more complex tasks (e.g., deep Q-learning or PPO for RL agents), we plan to parallelize the rollout generation. This will alleviate CPU bottlenecks that often limit the training speed and overall effectiveness of RL agents.
