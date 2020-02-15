# Author: Kaituo Xu, Fan Yu

def forward_algorithm(O, HMM_model):
    """HMM Forward Algorithm.
    Args:
        O: (o1, o2, ..., oT), observations
        HMM_model: (pi, A, B), (init state prob, transition prob, emitting prob)
    Return:
        prob: the probability of HMM_model generating O.
    """
    pi, A, B = HMM_model
    T = len(O)
    N = len(pi)
    prob = 0.0
    # Begin Assignment

    # Put Your Code Here
    alpha = [[0] * N for _ in range(T)]
    # 初值
    for i in range(N):
        alpha[0][i] = pi[i] * B[i][O[0]]

    # 递推
    for t in range(T - 1):
        for i in range(N):
            temp_value = 0
            for j in range(N):
                temp_value += alpha[t][j] * A[j][i]
            alpha[t + 1][i] = temp_value * B[i][O[t + 1]]

    # 终止
    for i in range(N):
        prob += alpha[-1][i]

    # End Assignment
    return prob


def backward_algorithm(O, HMM_model):
    """HMM Backward Algorithm.
    Args:
        O: (o1, o2, ..., oT), observations
        HMM_model: (pi, A, B), (init state prob, transition prob, emitting prob)
    Return:
        prob: the probability of HMM_model generating O.
    """
    pi, A, B = HMM_model
    T = len(O)
    N = len(pi)
    prob = 0.0
    # Begin Assignment

    # Put Your Code Here
    beta = [[0] * N for _ in range(T)]

    # 初值
    beta[-1][:] = [1] * N

    # 递归
    for t in range(T - 2, -1, -1):
        for i in range(N):
            temp_value = 0
            for j in range(N):
                temp_value += A[i][j] * B[j][O[t + 1]] * beta[t + 1][j]
            beta[t][i] = temp_value

    # 终止
    for i in range(N):
        prob += pi[i] * B[i][O[0]] * beta[0][i]

    # End Assignment
    return prob
 

def Viterbi_algorithm(O, HMM_model):
    """Viterbi decoding.
    Args:
        O: (o1, o2, ..., oT), observations
        HMM_model: (pi, A, B), (init state prob, transition prob, emitting prob)
    Returns:
        best_prob: the probability of the best state sequence
        best_path: the best state sequence
    """
    pi, A, B = HMM_model
    T = len(O)
    N = len(pi)
    best_prob, best_path = 0.0, []
    # Begin Assignment

    # Put Your Code Here
    delta = [[0] * N for _ in range(T)]
    phi = [[0] * N for _ in range(T)]

    # 初值
    for i in range(N):
        delta[0][i] = pi[i] * B[i][O[0]]
    # 递归
    for t in range(T - 1):
        for i in range(N):
            temp_value = [0] * N
            for j in range(N):
                temp_value[j] += delta[t][j] * A[j][i]
            delta[t + 1][i] = max(temp_value) * B[i][O[t + 1]]
            phi[t + 1][i] = temp_value.index(max(temp_value))

    # 终止
    best_prob = max(delta[-1][:])
    best_path.append(delta[-1][:].index(max(delta[-1][:])))

    for t in range(T - 1, 0, -1):
        end = best_path[-1]
        best_path.append(phi[t][end])

    # End Assignment
    return best_prob, best_path


if __name__ == "__main__":
    color2id = { "RED": 0, "WHITE": 1 }
    # model parameters
    pi = [0.2, 0.4, 0.4]
    A = [[0.5, 0.2, 0.3],
         [0.3, 0.5, 0.2],
         [0.2, 0.3, 0.5]]
    B = [[0.5, 0.5],
         [0.4, 0.6],
         [0.7, 0.3]]
    # input
    observations = (0, 1, 0)
    HMM_model = (pi, A, B)
    # process
    observ_prob_forward = forward_algorithm(observations, HMM_model)
    print(observ_prob_forward)

    observ_prob_backward = backward_algorithm(observations, HMM_model)
    print(observ_prob_backward)

    best_prob, best_path = Viterbi_algorithm(observations, HMM_model) 
    print(best_prob, best_path)
