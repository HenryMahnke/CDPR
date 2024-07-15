import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

def attachment_points_workspace(attachment_points, position):
    return attachment_points + position
# plot the payload and attachment points, stands, and vectors 
def plot(stands, attachment_points, position):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    attach = attachment_points_workspace(attachment_points, position)
    ax.scatter(stands[:,0], stands[:,1], stands[:,2], color='r', marker='^', s=100, label='Stand Positions')
    ax.scatter(attach[:,0], attach[:,1], attach[:,2], color='g', marker='o', s=50, label='Attachment Points')
    ax.scatter(position[0], position[1], position[2], color='b', marker='x', s=50, label='Payload Position')
    cables = find_cables(stands, attachment_points, position)
    for i in range(len(cables)):
        ax.plot([stands[i,0], stands[i,0]+cables[i,0]], 
                [stands[i,1], stands[i,1]+cables[i,1]], 
                [stands[i,2], stands[i,2]+cables[i,2]], 
                color='k', linestyle='--', linewidth=1)
    ax.set_zlim(0,stands[0,2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Payload and Attachment Points')
    ax.legend()
    plt.show()

def find_cables(stands,attachment_points, position):
    # direction of cables
    attach = attachment_points_workspace(attachment_points, position)
    cables =  attach- stands
    return cables




def calculate_cable_forces(stands, attachment_points,position,payload_mass, acceleration):
    # calculating cable unit vectors
    cables = find_cables(stands, attachment_points, position)
    attach_wrk = attachment_points_workspace(attachment_points, position)
    cable_lengths = np.linalg.norm(cables, axis=1)
    unit_vectors = cables / cable_lengths[:, np.newaxis]
    
    # setup the structure matrix, this represents how each cable contributes to forces and moments on the payload 
    A = np.zeros((6, len(stands))) # 6x4 matrix
    for i,unit_vector in enumerate(unit_vectors):
        A[:3,i] = unit_vector
        A[3:,i] = np.cross(attach_wrk[i]-position, unit_vector) #this finds momements or sum do more research
        
    # caluclate wrench
    weight = np.array([0,0,-payload_mass*9.81]) # cartesian forces
    interial_force = payload_mass * acceleration
    external_force = weight + interial_force # total force?
    external_moment = np.zeros(3)
    wrench = np.concatenate([external_force, external_moment])
    
    tensions = np.linalg.pinv(A) @ wrench
    forces = unit_vectors * tensions[:, np.newaxis]
    return forces, tensions

def calculate_cable_forces_linprog(stands, attachment_points, position, payload_mass, acceleration, max_tension):
    cables = find_cables(stands, attachment_points, position)
    attach_wrk = attachment_points_workspace(attachment_points, position)
    cable_lengths = np.linalg.norm(cables, axis=1)
    unit_vectors = cables / cable_lengths[:, np.newaxis]
    
    A = unit_vectors.T  # 3x4 matrix
    
    # Calculate external force (weight + inertial force)
    weight = np.array([0, 0, -payload_mass * 9.81])
    inertial_force = payload_mass * acceleration
    B = weight + inertial_force
    
    num_cables = len(stands)
    
    # Objective function: Minimize t
    c = np.hstack([np.zeros(num_cables), 1])
    
    # Inequality constraints: x_i <= t and -x_i <= 0 (must be positive for tension)
    A_ub = np.vstack([np.hstack([np.eye(num_cables), -np.ones((num_cables, 1))]),
                      np.hstack([-np.eye(num_cables), np.zeros((num_cables, 1))])])
    b_ub = np.zeros(2 * num_cables)
    
    # Equality constraints: Ax = B
    A_eq = np.hstack([A, np.zeros((A.shape[0], 1))])
    b_eq = B
    
    # Bounds for variables
    bounds = [(0, max_tension) for _ in range(num_cables)] + [(0, None)]  # Last bound is for t
    
    # Solve the linear programming problem
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if res.success:
        tensions = res.x[:num_cables]
        return tensions
    else:
        return None

def find_workspace(stands,attachment_points,acceleration,max_tension,min_tension,payload_mass):
    x = np.linspace(0, 5, 20)
    y = np.linspace(0, 5, 20)
    z = np.linspace(0, 3, 20)
    feasible_workspace = []
    for xi in x:
        for yi in y:
            for zi in z:
                pos = np.array([xi, yi, zi])
                tensions = calculate_cable_forces_linprog(stands, attachment_points, pos, payload_mass, acceleration, max_tension)
                if tensions is not None:
                    feasible_workspace.append(pos)

    feasible_workspace = np.array(feasible_workspace)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(feasible_workspace[:, 0], feasible_workspace[:, 1], feasible_workspace[:, 2], 
                   color='b', marker='o', s=2, alpha=0.6)
    ax.scatter(stands[:,0], stands[:,1], stands[:,2], color='r', marker='^', s=100, label='Stand Positions')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Feasible Workspace')
    plt.show()
    
# should change to use linprog for finding tension 

payload_mass = 1 # kg
payload_size = 0.5 # meters
robot_size = [5,5,3] # meters
stands = np.array([
    [0,0,3],
    [5,0,3],
    [0,5,3],
    [5,5,3]
])
# relative to payload size
attachment_points = np.array([
    [-0.25,-0.25,0.25],
    [0.25,-0.25,0.25],
    [-0.25,0.25,0.25],
    [0.25,0.25,0.25]
])
# acceleration vector of the payload
acceleration = np.array([0,0,0]) # gravity already accounted for
position = np.array([2.5,2.5,2.5]) # meters
max_motor_torque = 1.5 # Newton meters
radius_drums = 0.1 # meters
max_cable_tension = max_motor_torque / radius_drums
print(max_cable_tension)
min_tension = 0 # Newtons
max_cable_length = 10
min_cable_length = 0.1


# calculate the direction vector
# print(calculate_cable_forces(stands, attachment_points, position, payload_mass, acceleration))
# plot(stands, attachment_points, position)
# print(find_cables(stands, attachment_points, position))
find_workspace(stands, attachment_points, acceleration, max_cable_tension, min_tension, payload_mass)