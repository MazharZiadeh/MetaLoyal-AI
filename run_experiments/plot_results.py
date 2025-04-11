import matplotlib.pyplot as plt

# Load logs
with open("results/reward_log.txt", "r") as f:
    rewards = list(map(float, f.read().split(",")))

with open("results/loyalty_log.txt", "r") as f:
    loyalties = list(map(float, f.read().split(",")))

with open("results/betrayal_log.txt", "r") as f:
    betrayals = list(map(int, f.read().split(",")))

# Plotting
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(rewards, label="Reward")
plt.plot(loyalties, label="Loyalty Score")
plt.xlabel("Step")
plt.ylabel("Value")
plt.title("Reward vs Loyalty Score")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(betrayals, "r.", label="Betrayal Events")
plt.xlabel("Step")
plt.ylabel("Betrayed (1=True)")
plt.title("Betrayal Timeline")
plt.legend()

plt.tight_layout()
plt.savefig("results/reward_vs_loyalty.png")
plt.show()
