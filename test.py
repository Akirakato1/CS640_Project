import numpy as np

states = [0, 1, 2]
actions = [0, 1]  # 0 : stay, 1 : jump
jump_probabilities = np.matrix([[0.1, 0.2, 0.7],
                                [0.5, 0, 0.5],
                                [0.6, 0.4, 0]])
for i in range(len(states)):
    jump_probabilities[i, :] /= jump_probabilities[i, :].sum()

rewards_stay = np.array([0, 8, 5])
rewards_jump = np.matrix([[-5, 5, 7],
                          [2, -4, 0],
                          [-3, 3, -3]])

T = np.zeros((len(states), len(actions), len(states)))
R = np.zeros((len(states), len(actions), len(states)))
for s in states:
    T[s, 0, s], R[s, 0, s] = 1, rewards_stay[s]
    T[s, 1, :], R[s, 1, :] = jump_probabilities[s, :], rewards_jump[s, :]

example_1 = (states, actions, T, R)

state = np.max(np.sum(T[0, :, :] * R[0, :, :], axis=1))
state1 = np.sum(T[1, :, :] * R[1, :, :], axis=1)
state2 = np.sum(T[2, :, :] * R[2, :, :], axis=1)
states = np.sum(T * R,axis=2)


print(state)
print(state1)
print(state2)
print(states)


def value_iteration(states, actions, T, R, gamma=0.1, tolerance=1e-2, max_steps=100):
    Vs = []  # all state values
    Vs.append(np.zeros(len(states)))  # initial state values
    steps, convergent = 0, False
    while not convergent:
        ########################################################################
        # TO DO: compute state values, and append it to the list Vs
        qv = np.sum(T * (R + gamma * Vs[-1]), axis=2)
        state = np.max(qv, axis=1)
        Vs.append(state)
        ############################ End of your code ##########################
        steps += 1
        convergent = np.linalg.norm(Vs[-1] - Vs[-2]) < tolerance or steps >= max_steps
    ########################################################################
    # TO DO: extract policy and name it "policy" to return
    policy = [actions[np.argmax(qv, axis=1)[i]] for i in range(len(state))]
    # for idx,state in enumerate(Vs):
    #     pol=[actions[np.argmax(np.sum(T*(R+gamma*state),axis=2),axis=1)[i]] for i in range(len(state))]
    #     policy.append(pol)
    ############################ End of your code ##########################
    return Vs, policy, steps


print("Example MDP 1")
states, actions, T, R = example_1
gamma, tolerance, max_steps = 0.1, 1e-2, 100
Vs, policy, steps = value_iteration(states, actions, T, R, gamma, tolerance, max_steps)
for i in range(steps):
    print("Step " + str(i))
    print("state values: " + str(Vs[i]))
    print()
print("Optimal policy: " + str(policy))


def policy_iteration(states, actions, T, R, gamma=0.1, tolerance=1e-2, max_steps=100):
    policy_list = []  # all policies explored
    initial_policy = np.array([np.random.choice(actions) for s in states])  # random policy
    policy_list.append(initial_policy)
    Vs = []  # all state values
    Vs.append(np.zeros(len(states)))  # initial state values
    steps, convergent = 0, 0
    while not convergent:
        ########################################################################
        # TO DO:
        # 1. Evaluate the current policy, and append the state values to the list Vs
        c = 0
        while not c:
            qv = np.sum(T * (R + gamma * Vs[-1]), axis=2)
            state = np.array([qv[i][policy_list[-1][i]] for i in range(len(states))])
            Vs.append(state)
            c = np.linalg.norm(Vs[-1] - Vs[-2]) < tolerance

        # 2. Extract the new policy, and append the new policy to the list policy_list
        policy = np.array([actions[np.argmax(np.sum(T * (R + gamma * Vs[-1]), axis=2), axis=1)[i]] for i in range(len(Vs[-1]))])
        policy_list.append(policy)
        ############################ End of your code ##########################
        steps += 1
        convergent = (policy_list[-1] == policy_list[-2]).all() or steps >= max_steps
    return Vs, policy_list, steps


print("Example MDP 1")
states, actions, T, R = example_1
gamma, tolerance, max_steps = 0.1, 1e-2, 100
Vs, policy_list, steps = policy_iteration(states, actions, T, R, gamma, tolerance, max_steps)
for i in range(steps):
    print("Step " + str(i))
    print("state values: " + str(Vs[i]))
    print("policy: " + str(policy_list[i]))
    print()
print()
