import numpy as np

def line_intersection(p1, p2, p3, p4):
    # Direction vectors
    line1 = np.linspace(p1,p2,100)
    line2 = np.linspace(p3,p4,100)
    n = 100
    threshold = 0.1
    for i in range(n):
        for j in range(i+1, n):
            if np.all(abs(line1[i] - line2[j]) <= threshold):
                return True
    return False


# Example usage
line_intersection(np.array([0, 0, 0]), np.array([1, 1, 1]), np.array([0, 1, 0]), np.array([1, 0, 1]))