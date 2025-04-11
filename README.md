# MetaLoyal: Learning Human-Like Loyalty in Reinforcement Learning Agents

MetaLoyal is an experimental framework to train AI agents to exhibit *loyalty*—a persistent commitment to another agent or value system—even when faced with short-term gain from betrayal.

## 🧠 Project Overview
This project implements a custom environment (`TemptationBridgeEnv`) where agents must choose between loyalty and betrayal, receiving shaped rewards based on their loyalty score.

We train RL agents using `Stable-Baselines3` and analyze their emergent behavior under different loyalty shaping parameters.

## 📂 Project Structure

- `environment/`: Custom PettingZoo-compatible environment
- `training/`: Scripts to train and evaluate agents
- `analysis/`: Tools for plotting and behavior analysis
- `results/`: Pretrained models and graph outputs
- `paper/`: Research paper draft and citations
- `utils/`: Helper wrappers and utilities
