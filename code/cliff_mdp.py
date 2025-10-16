import numpy as np
import random
import copy


# by default, actions are ordered
# [left, up, down, right]
def get_cliff_mdp(state_list, state_rowcol, rowcol_state, grid):
    

    # --- build cliff MDP ---
    """
    Returns a dict with:
      S: list of states (length 38)
      A: list of actions in fixed order [left, up, down, right]
      P: transition probabilities, shape (4, 38, 38) with order (a, s_idx, sprime_idx)
      R: rewards aligned with P, shape (4, 38, 38)
      s_index: mapping from state-id -> index 0..37
      index_s: inverse mapping
    Dynamics:
      - deterministic grid moves except:
        * stepping into any cliff cell (bottom row columns 1..10) => teleports to start state 1 with reward -100
        * goal state 2 is absorbing with reward 0
      - off-grid attempts: remain in place with reward -1
      - all other valid moves: reward -1
    """

    S = state_list[:]
    A = ["left", "up", "down", "right"]
    num_S = len(S)
    s_index = {s: i for i, s in enumerate(S)}
    index_s = {i: s for i, s in enumerate(S)}
    
    
    start_state = 1
    goal_state = 2
    # cliff_states = [c for c in range(1, 11)]
    # print("cliff_states:", cliff_states)
    
    # initial state distribution
    eta = np.zeros(num_S)
    eta[s_index[start_state]] = 1.0

    P = np.zeros((num_S, len(A), num_S))
    R = np.zeros((num_S, len(A), num_S))
    
    for s in S:
        s_idx = s_index[s]
        r, c = state_rowcol[s]
        for a_idx, a in enumerate(A):
            if s == goal_state:
                P[s_idx, a_idx, s_idx] = 1.0
                R[s_idx, a_idx, s_idx] = 0.0
                continue
            
            next_r, next_c = r, c  # default: stay in place
            reward = -1.0  # default reward for valid move
            
            if a == "left":
                temp_r, temp_c = r, c - 1
                if 0 <= temp_r < 4 and 0 <= temp_c < 12 and grid[temp_r][temp_c] is not None:
                    next_r, next_c = temp_r, temp_c
                else:
                    next_r, next_c = r, c  # off-grid/bump into wall, stay in place
            elif a == "up":
                temp_r, temp_c = r - 1, c
                if 0 <= temp_r < 4 and 0 <= temp_c < 12 and grid[temp_r][temp_c] is not None:
                    next_r, next_c = temp_r, temp_c
                else:
                    next_r, next_c = r, c  # off-grid/bump into wall, stay in place
            elif a == "down":
                temp_r, temp_c = r + 1, c
                if r == 2:  # moving down into cliff row
                    if temp_c == 11:  # goal state column
                        next_r, next_c = state_rowcol[goal_state]
                        reward = 0.0
                    elif 1 <= temp_c <= 10:  # cliff columns
                        next_r, next_c = state_rowcol[start_state]
                        reward = -100.0
                    elif temp_c == 0:  # left wall
                        next_r, next_c = r, c  # stay in place
                        reward = -1.0
                    else: 
                        next_r, next_c = r, c  # stay in place
                        reward = -1.0
                else:
                    if 0 <= temp_r < 4 and 0 <= temp_c < 12 and grid[temp_r][temp_c] is not None:
                        next_r, next_c = temp_r, temp_c
                    else:
                        next_r, next_c = r, c  # off-grid/bump into wall, stay in place
            elif a == "right":
                temp_r, temp_c = r, c + 1
                if r == 3 and c == 0 and 0 <= temp_r < 4 and 0 <= temp_c < 12:
                    next_r, next_c = state_rowcol[start_state]
                    reward = -100.0
                elif 0 <= temp_r < 4 and 0 <= temp_c < 12 and grid[temp_r][temp_c] is not None:
                    next_r, next_c = temp_r, temp_c
                else:
                    next_r, next_c = r, c  # off-grid/bump into wall, stay in place
            
            next_s = rowcol_state[(next_r, next_c)]
            next_s_idx = s_index[next_s]
            P[s_idx, a_idx, next_s_idx] = 1.0
            R[s_idx, a_idx, next_s_idx] = reward

    return {"S": S, "A": A, "P": P, "R": R, "s_index": s_index, "index_s": index_s, "eta": eta, "start": start_state, "goal": goal_state}


# by default, actions are ordered
# [left, up, down, right]
def get_optimal_policy(state_list, state_rowcol, rowcol_state, grid):
    mdp = get_cliff_mdp(state_list, state_rowcol, rowcol_state, grid)
    pi = np.full(len(state_list), -1, dtype=int)  # -1 means undefined
    s_index = mdp["s_index"]
    
    print("Optimal policy (0=left,1=up,2=down,3=right):")
    path = [(3,0),(2,0)]
    path += [(2,c) for c in range(1,12)]
    path += [(3,11)]
    for (r,c), (nr,nc) in zip(path[:-1], path[1:]):
        s = rowcol_state[(r,c)]
        if   (nr,nc) == (r,   c-1): a = 0
        elif (nr,nc) == (r-1, c  ): a = 1
        elif (nr,nc) == (r+1, c  ): a = 2
        else:                        a = 3
        pi[s_index[s]] = a
        print(f" --> state {s} at ({r},{c}) -> action {a}")
    
    return pi


def get_pi_e_greedy(pi, epsilon):
    num_S = len(pi)
    num_A = 4  # fixed number of actions [left, up, down, right]
    pi_epsilon = np.zeros((num_S, num_A))
    print(f"Constructing epsilon-greedy policy with epsilon={epsilon}")
    for s in range(num_S):
        for a in range(num_A):
            if a == pi[s]:
                pi_epsilon[s, a] = 1.0 - epsilon
            else:
                pi_epsilon[s, a] = epsilon
    print("Epsilon-greedy policy matrix (rows=states, cols=actions):")
    print(pi_epsilon)
    
    return pi_epsilon

# Given an MDP and a policy pi, return the induced Markov chain's
# transition matrix P_pi and reward vector R_pi.
# P_pi has shape (S, S) and R_pi has shape (S,)
def get_P_R(MDP, pi):
    P, R = MDP["P"], MDP["R"]
    S = P.shape[0]
    
    # If pi is given as a 1D array of action indices, convert to one-hot
    if pi.ndim == 1:
        policy_matrix = np.zeros((S, 4))
        # Convert pi to one-hot encoding
        policy_matrix[np.arange(S), pi] = 1.0
        pi = policy_matrix
    
    # P_pi(s,s') = sum_a pi(s,a) * P(s,a,s')
    P_pi = (pi[:, :, None] * P).sum(axis=1)

    # Expected immediate reward per (s,a): r_sa = sum_{s'} P * R
    R_sa = (P * R).sum(axis=2)

    # R_pi(s) = sum_a pi(s,a) * r_sa(s,a)
    R_pi = (pi * R_sa).sum(axis=1)

    return P_pi, R_pi

def get_v(P, R, gamma):
    # Solve v = (I - gamma * P_pi)^{-1} * R_pi
    I = np.eye(P.shape[0])
    A = I - gamma * P
    
    return np.linalg.solve(A, R)


# Generate a full trajectory under the policy pi.
# The trajectory must start in init_state and end in terminal
def gen_episode(MDP, pi, num_actions, init_state, terminal):
    P, R = MDP["P"], MDP["R"]
    s_index = MDP["s_index"]; index_s = MDP["index_s"]
    traj = []
    s = init_state
    t = 0
    print(pi)
    
    while s != terminal and t < num_actions:
        i = s_index[s]
        if pi.ndim == 1:
            a_probs = pi[i]  # deterministic policy
        else:
            a_probs = np.random.choice(4, p=pi[i])  # stochastic policy
        
        # print(f"state {s} at {MDP['s_index'][s]}: action probs {a_probs}")
        # print(f"  taking action {a_probs}")

        probs = P[a_probs, i]
        # print(f"  next state probs: {probs}")

        next_s_idx = int(np.random.choice(len(s_index), p=probs))
        next_s = index_s[next_s_idx]

        r = float(R[a_probs, i, next_s_idx])

        traj.append((s, a_probs, r, next_s))
        
        s = next_s
        
        t += 1
    # print(f"Terminal state {s} reached after {t} steps.")
    # print("Trajectory:")
    # print(traj)
    return traj


if __name__ == "__main__":

    random.seed(1)

    gamma = 0.9

    # YOUR CODE GOES HERE
    
    # --- build the grid ---
    # Columns: 0..11 (left->right)
    # Rows:    0..3  (top->bottom)
    # States are:
    #   top 3 rows fully:   row0: 3,6,9,...,36 ; row1: 4,7,10,...,37 ; row2: 5,8,11,...,38
    #   bottom corners only: (row3,col0)=1  (row3,col11)=2
    grid = [[None for _ in range(12)] for _ in range(4)]
    for c in range(12):
        grid[0][c] = 3 + 3*c
        grid[1][c] = 4 + 3*c
        grid[2][c] = 5 + 3*c
    grid[3][0] = 1
    grid[3][11] = 2
    print("grid:")
    for r in range(4):
        print(grid[r])
        
    state_list = grid[3][:] + [grid[r][c] for r in range(3) for c in range(12)]
    state_list = sorted([s for s in state_list if s is not None])
    
    # mappings between state-id and (row,col)
    state_rowcol = {s: (r, c) for r in range(4) for c in range(12) if (s := grid[r][c]) is not None}
    rowcol_state = {(r, c): s for r in range(4) for c in range(12) if (s := grid[r][c]) is not None}
    
    cliff_mdp = get_cliff_mdp(state_list, state_rowcol, rowcol_state, grid)

    pi = get_optimal_policy(state_list, state_rowcol, rowcol_state, grid)
    print("Optimal policy array:", pi)

    # epsilon-greedy policies
    pi_01 = get_pi_e_greedy(pi, 0.1)
    pi_02 = get_pi_e_greedy(pi, 0.2)
    

    # get infinite-horizon value function for optimal policy
    P_pi, R_pi = get_P_R(cliff_mdp, pi)
    v = get_v(P_pi, R_pi, gamma)
    
    print("Optimal policy value function:")
    for s in cliff_mdp["S"]:
        print(f"  state {s:2d} at {state_rowcol[s]}: v={v[cliff_mdp['s_index'][s]]:.2f}")
        
    
    # get infinite-horizon value function for epsilon-greedy policies
    v_01 = get_v(*get_P_R(cliff_mdp, pi_01), gamma)
    v_02 = get_v(*get_P_R(cliff_mdp, pi_02), gamma)
    
    print("Epsilon=0.1 policy value function:")
    for s in cliff_mdp["S"]:
        print(f" --> state {s:2d} at {state_rowcol[s]}: v={v_01[cliff_mdp['s_index'][s]]:.2f}")

    print("Epsilon=0.2 policy value function:")
    for s in cliff_mdp["S"]:
        print(f" --> state {s:2d} at {state_rowcol[s]}: v={v_02[cliff_mdp['s_index'][s]]:.2f}")
    
    for _ in range(1000):
        # print("\n--- Generating episode with optimal policy ---")
        traj = gen_episode(cliff_mdp, pi, 1000, cliff_mdp["start"], cliff_mdp["goal"])