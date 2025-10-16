import numpy as np
import random
import copy


# by default, actions are ordered
# [left, up, down, right]
def get_cliff_mdp(state_list, state_rowcol, rowcol_state, grid):
    # --- build cliff MDP ---
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
    # eta = np.zeros(num_S)
    # eta[s_index[start_state]] = 1.0

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
                temp_c = c - 1
                if 0 <= r < 4 and 0 <= temp_c < 12 and grid[r][temp_c] is not None:
                    next_r, next_c = r, temp_c
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

    return {"S": S, "A": A, "P": P, "R": R, "s_index": s_index, "index_s": index_s, "start": start_state, "goal": goal_state}


# by default, actions are ordered
# [left, up, down, right]
def get_optimal_policy(state_list, state_rowcol, rowcol_state, grid):
    mdp = get_cliff_mdp(state_list, state_rowcol, rowcol_state, grid)
    # pi = np.full(len(mdp["S"]), -1, dtype=int)  # -1 means undefined
    pi = np.zeros(len(mdp["S"]), dtype=int)
    s_index = mdp["s_index"]
    
    print(" \nOptimal policy (0=left,1=up,2=down,3=right):")
    path = [(3,0),(2,0)]
    path += [(2,c) for c in range(1,12)]
    path += [(3,11)]
    
    for i in range(len(path)-1):
        r, c = path[i]
        next_r, next_c = path[i+1]
        s = rowcol_state[(r,c)]
        if (next_r, next_c) == (r, c-1):
            a = 0  # left
        elif (next_r, next_c) == (r-1, c):
            a = 1  # up
        elif (next_r, next_c) == (r+1, c):
            a = 2  # down
        elif (next_r, next_c) == (r, c+1):
            a = 3  # right
        
        pi[s_index[s]] = a
        print(f"state {s} at ({r},{c}) -> action {a} ==> ", end="")
    
    return pi


def get_pi_e_greedy(pi, epsilon):
    num_S = len(pi)
    num_A = 4  # fixed number of actions [left, up, down, right]
    pi_epsilon = np.zeros((num_S, num_A))
    print(f"Constructing epsilon-greedy policy with epsilon={epsilon}")
    for s in range(num_S):
        for a in range(num_A):
            if a == pi[s]:
                pi_epsilon[s, a] = 1.0 - epsilon + (epsilon / num_A)
            else:
                pi_epsilon[s, a] = epsilon / num_A
    
    return pi_epsilon


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
    trajectory = []
    
    s = init_state
    t_steps = 0

    while s != terminal and t_steps < num_actions:
        idx = s_index[s]
        if pi.ndim == 1:
            action = int(pi[idx])  # deterministic policy pi
            # print(f"state {s} at {idx}: taking action {action} with probs {pi[idx]} -> ", end="")
        else:
            action = int(np.random.choice(4, p=pi[idx]))  # stochastic policy pi (0.1-epsilon greedy/0.2-epsilon greedy)
            # print(f"state {s} at {idx}: taking action {action} with probs {pi[idx, action]} -> ", end="")

        # probs = P[idx, action]        
        # Sample next state based on transition probabilities
        next_s_idx = int(np.random.choice(len(s_index), p=P[idx, action]))
        next_s = index_s[next_s_idx]
        
        r = float(R[idx, action, next_s_idx])
        
        trajectory.append((s, action, r, next_s))

        s = next_s
        t_steps += 1

    # print(f"\nGenerated trajectory (length {len(trajectory)}):")
    # print(trajectory)

    return trajectory


if __name__ == "__main__":

    random.seed(1)

    gamma = 0.9

    # YOUR CODE GOES HERE
    
    # --- build the grid ---
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
    print(f"\nOptimal policy array: {pi}\n" )

    # epsilon-greedy policies
    pi_01 = get_pi_e_greedy(pi, 0.1)
    pi_02 = get_pi_e_greedy(pi, 0.2)
    

    # get infinite-horizon value function for optimal policy
    P_pi, R_pi = get_P_R(cliff_mdp, pi)
    v = get_v(P_pi, R_pi, gamma)
    
    # get infinite-horizon value function for epsilon-greedy policies
    v_01 = get_v(*get_P_R(cliff_mdp, pi_01), gamma)
    v_02 = get_v(*get_P_R(cliff_mdp, pi_02), gamma)
    
    state_values = list(zip(cliff_mdp["S"], v, v_01, v_02))
    state_values.sort(key=lambda x: x[0])

    # Print header
    print(f"\nCliff: State values for pi, pi_0.1, pi_0.2, gamma={gamma}")
    print("-" * 70)
    print(f"{'State':<10} {'V_pi':<20} {'V_pi_0.1':<20} {'V_pi_0.2':<20}")
    print("-" * 70)

    # Print each row
    for state, v_opt, v_e01, v_e02 in state_values:
        print(f"{state:<10} {v_opt:<20.4f} {v_e01:<20.4f} {v_e02:<20.4f}")

    print("-" * 70)
    
    # get average discounted returns over 1000 episodes
    num_episodes = 1000
    G_pi_total = 0.0
    G_pi_01_total = 0.0
    G_pi_02_total = 0.0
    
    print(f"\nRunning {num_episodes} episodes for each policy...")
    for ep in range(num_episodes):
        traj_pi = gen_episode(cliff_mdp, pi, 1000, cliff_mdp["start"], cliff_mdp["goal"])
        traj_pi_01 = gen_episode(cliff_mdp, pi_01, 1000, cliff_mdp["start"], cliff_mdp["goal"])
        traj_pi_02 = gen_episode(cliff_mdp, pi_02, 1000, cliff_mdp["start"], cliff_mdp["goal"])
        
        # discounted return G = sum_t (gamma^t * r_t) for each trajectory
        for t, (s, a, r, s_next) in enumerate(traj_pi):
            G_pi_total += (gamma ** t) * r

        for t, (s, a, r, s_next) in enumerate(traj_pi_01):
            G_pi_01_total += (gamma ** t) * r

        for t, (s, a, r, s_next) in enumerate(traj_pi_02):
            G_pi_02_total += (gamma ** t) * r
            
    G_avg_pi = G_pi_total / num_episodes
    G_avg_pi_01 = G_pi_01_total / num_episodes
    G_avg_pi_02 = G_pi_02_total / num_episodes
    
    print("\nAverage Discounted Returns over 1000 episodes:")
    print("-" * 60)
    print(f"{'Policy':<30} {'Average G':<20}")
    print("-" * 60)
    print(f"{'pi (optimal)':<30} {G_avg_pi:<20.4f}")
    print(f"{'pi_0.1 (epsilon=0.1)':<30} {G_avg_pi_01:<20.4f}")
    print(f"{'pi_0.2 (epsilon=0.2)':<30} {G_avg_pi_02:<20.4f}")
    print("-" * 60)
    