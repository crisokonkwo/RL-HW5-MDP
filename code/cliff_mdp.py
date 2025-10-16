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
    pi = np.zeros(len(mdp["S"]), dtype=int)
    print("Optimal policy (0=left,1=up,2=down,3=right):")
    path = [(3,0),(2,0)]
    path += [(2,c) for c in range(1,12)]
    path += [(3,11)]
    for i in range(len(path)-1):
        r,c = path[i]
        next_r, next_c = path[i+1]
        s = rowcol_state[(r,c)]
        s_idx = mdp["s_index"][s]
        if next_r == r and next_c == c-1:
            a = 0
        elif next_r == r-1 and next_c == c:
            a = 1
        elif next_r == r+1 and next_c == c:
            a = 2
        elif next_r == r and next_c == c+1:
            a = 3
        else:
            raise ValueError("Invalid path step")
        pi[s_idx] = a
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
                pi_epsilon[s, a] = 1.0 - epsilon + (epsilon / num_A)
            else:
                pi_epsilon[s, a] = (epsilon / num_A)
    
    print(pi_epsilon)
    return pi_epsilon

# Given an MDP and a policy pi, return the induced Markov chain's
# transition matrix P_pi and reward vector R_pi.
# P_pi has shape (S, S) and R_pi has shape (S,)
def get_P_R(MDP, pi):
    P, R = MDP["P"], MDP["R"]
    S = P.shape[1]
    P_pi = np.zeros((S, S))
    R_pi = np.zeros(S)
    for s in range(S):
        a = pi[s]
        P_pi[s, :] = P[s, a, :]
        R_pi[s] = R[s, a, :].dot(P[s, a, :])
    
    return P_pi, R_pi

def get_v(P, R, gamma):
    I = np.eye(P.shape[0])
    A = I - gamma * P
    
    return np.linalg.solve(A, R)


# Generate a full trajectory under the policy pi.
# The trajectory must start in init_state and end in terminal
def gen_episode(MDP, pi, num_actions, init_state, terminal):
    pass


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

    # epsilon-greedy policies
    pi_01 = get_pi_e_greedy(pi, 0.1)
    pi_02 = get_pi_e_greedy(pi, 0.2)