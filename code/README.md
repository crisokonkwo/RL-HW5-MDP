# Cliff Walking - Markov Decision Processes (MDP)

This project implements a MDP for the classic Cliff Walking problem. The agent must navigate from a start state to a goal state while avoiding a cliff that results in a large negative reward.

## Grid World Layout

- **Grid Size**: 4 rows × 12 columns
- **Start State**: Bottom-left corner (state 1)
- **Goal State**: Bottom-right corner (state 2)
- **Cliff**: Bottom row, columns 1-10 (states between start and goal)
- **Safe Path**: Through the upper three rows

## Rewards

- **Normal move**: -1
- **Falling off cliff**: -100 (returns to start)
- **Reaching goal**: 0

## Actions

- **0**: Left
- **1**: Up
- **2**: Down
- **3**: Right

## Results

The code outputs:

1. **State Value Table**: Shows V values for all states under each policy
2. **Average Discounted Returns**: Compares empirical performance across policies

### Expected Observations

- Optimal policy achieves highest average return
- $\epsilon$-greedy policies have lower returns due to exploration
- Higher $\epsilon$ leads to more exploration and potentially lower returns with 10% or 20% randomness
- Value functions reflect the safety-reward tradeoff i.e. the expected discounted return

$$ V_{\pi}(s) = (I - \gamma P_{\pi})^{-1}R_{\pi}(s)$$

- Average discounted return over 1000 episodes

$$G = \frac{1}{1000} \sum_{t=0}^{T-1} \gamma^{t}r_t$$

## Usage

```python
python cliff_mdp.py
```

### Key Parameters

- **gamma $(\gamma)$**: 0.9 - Discount factor for future rewards
- **num_episodes**: 1000 - Number of episodes for averaging
- **epsilon $(\epsilon)$**: 0.1, 0.2 - Exploration rates for $\epsilon$-greedy policies
  