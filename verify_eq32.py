import numpy as np

def verify_eq32():
    print("Verifying Eq 3.2 on a K_{2,2} bipartite graph.")
    # Adjacency matrix for K_{2,2}
    # Nodes 0, 1 on one side; Nodes 2, 3 on the other side.
    # Edges: (0,2), (0,3), (1,2), (1,3)
    A = np.array([
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [1, 1, 0, 0]
    ])
    
    A2 = A.dot(A)
    A3 = A2.dot(A)
    
    print("Adjacency Matrix A:\n", A)
    print("A^3 Matrix:\n", A3)
    
    # Consider edge e = (0, 2)
    u, v = 0, 2
    A_uv = A[u, v]
    d_u = np.sum(A[u, :])
    d_v = np.sum(A[v, :])
    
    # By Eq 3.2:
    sigma_C4_eq = A_uv * A3[u, v] - d_u - d_v + 1
    
    # Brute force 4-cycle count for edge (0,2):
    # The only 4-cycle containing (0,2) is 0 -> 2 -> 1 -> 3 -> 0
    # There is exactly 1 such cycle.
    
    print(f"Edge ({u}, {v}):")
    print(f"  A_{u}{v} = {A_uv}")
    print(f"  (A^3)_{u}{v} = {A3[u, v]}")
    print(f"  d({u}) = {d_u}, d({v}) = {d_v}")
    print(f"  Calculated Sigma_C4 = {sigma_C4_eq}")
    print(f"  Expected Sigma_C4 = 1")
    
    assert sigma_C4_eq == 1, "Verification failed!"
    print("Verification successful: Eq 3.2 matches brute-force counting.")

if __name__ == "__main__":
    verify_eq32()
