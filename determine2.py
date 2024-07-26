import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

class CDPR:
    def __init__(self, stands, attachment_points, payload_mass, payload_size, robot_size,
                 max_motor_torque_oz_in, gear_ratio, motor_rpm, drum_radius,spool_length_m,cable_diameter_m):
        self.stands = stands
        self.attachment_points = attachment_points
        self.payload_mass = payload_mass
        self.payload_size = payload_size
        self.robot_size = robot_size
        
        # Motor and gearing calculations
        self.max_motor_torque = max_motor_torque_oz_in * 0.00706155  # Convert oz-in to N-m
        self.gear_ratio = gear_ratio
        self.motor_rpm = motor_rpm
        self.drum_radius = drum_radius
        
        self.drum_rotation_rate = self.motor_rpm / self.gear_ratio
        self.max_payload_speed = self.drum_rotation_rate * self.drum_radius * 2 * np.pi / 60
        self.max_torque = self.max_motor_torque * self.gear_ratio
        self.max_cable_tension = self.max_torque / self.drum_radius
        self.min_tension = 0  # Newtons
        self.max_tension = self.max_cable_tension
        self.spool_length_m = spool_length_m
        self.cable_diameter_m = cable_diameter_m
        
        print(f"Max Payload Speed: {self.max_payload_speed:.2f} m/s")
        print(f"Max Cable Tension: {self.max_cable_tension:.2f} N")

    def attachment_points_workspace(self, position):
        return self.attachment_points + position

    def find_cables(self, position):
        attach = self.attachment_points_workspace(position)
        cables = attach - self.stands
        
        return cables
    def find_cable_collisions(self, position):
        attach = self.attachment_points_workspace(position)
        collisions = []

        if len(attach) < 2:
            return collisions  # No collisions possible with fewer than 2 cables

        for i in range(len(attach)):
            for j in range(i + 1, len(attach)):
                if self.line_intersection(self.stands[i], attach[i], self.stands[j], attach[j]):
                    collisions.append((i, j))

        return collisions
    def line_intersection(self,p1, p2, p3, p4):
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

    def plot(self, position):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        attach = self.attachment_points_workspace(position)
        ax.scatter(self.stands[:,0], self.stands[:,1], self.stands[:,2], color='r', marker='^', s=100, label='Stand Positions')
        ax.scatter(attach[:,0], attach[:,1], attach[:,2], color='g', marker='o', s=50, label='Attachment Points')
        ax.scatter(position[0], position[1], position[2], color='b', marker='x', s=50, label='Payload Position')
        cables = self.find_cables(position)
        for i in range(len(cables)):
            ax.plot([self.stands[i,0], self.stands[i,0]+cables[i,0]], 
                    [self.stands[i,1], self.stands[i,1]+cables[i,1]], 
                    [self.stands[i,2], self.stands[i,2]+cables[i,2]], 
                    color='k', linestyle='--', linewidth=1)
        ax.set_zlim(0, self.stands[0,2])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Payload and Attachment Points')
        ax.legend()
        plt.show()

    def calculate_cable_forces_linprog(self, position, acceleration):
        cables = self.find_cables(position)
        attach_wrk = self.attachment_points_workspace(position)
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
        bounds = [(0, self.max_cable_tension) for _ in range(num_cables)] + [(0, None)]  # Last bound is for t
        
        # Solve the linear programming problem
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        
        if res.success:
            tensions = res.x[:num_cables]
            return tensions
        else:
            return None

    def find_workspace(self, acceleration, resolution=20):
        x = np.linspace(0, self.robot_size[0], resolution)
        y = np.linspace(0, self.robot_size[1], resolution)
        z = np.linspace(0, self.robot_size[2], resolution)
        feasible_workspace = []
        tension_values = []
        for xi in x:
            for yi in y:
                for zi in z:
                    pos = np.array([xi, yi, zi])
                    tensions = self.calculate_cable_forces_linprog(pos, acceleration)
                    if tensions is not None and np.all(tensions >= self.min_tension) and np.all(tensions <= self.max_cable_tension):
                        feasible_workspace.append(pos)
                        tension_values.append(np.max(tensions))

        if not feasible_workspace:
            print("No feasible workspace found with the given parameters.")
            return

        feasible_workspace = np.array(feasible_workspace)
        tension_values = np.array(tension_values)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(feasible_workspace[:, 0], feasible_workspace[:, 1], feasible_workspace[:, 2], 
                       c=tension_values, cmap='viridis', marker='o', s=10, alpha=0.8)
        ax.scatter(self.stands[:,0], self.stands[:,1], self.stands[:,2], color='r', marker='^', s=100, label='Stand Positions')
        cbar = fig.colorbar(scatter)
        cbar.set_label('Max Tension (N)', rotation=270, alpha=0.8)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Feasible Workspace with mass {self.payload_mass} kg')
        plt.show()
    def determineCharacteristics(self):
        # finding max tension
        # finding max speed
        # finding max acceleration
        # finding cable length
        num_turns = self.spool_length_m / self.cable_diameter_m 
        total_cable_length_m = num_turns * 2 * np.pi * self.drum_radius #meters
        total_cable_length_ft = total_cable_length_m * 3.28084 #feet
        print(f"Total Cable Length: {total_cable_length_m:.2f} m ({total_cable_length_ft:.2f} ft)")
        pass
        

# Example usage:

        
stands = np.array([
    [0,0,3],
    [5,0,3],
    [0,5,3],
    [5,5,3]
])
attachment_points = np.array([
    [-0.25,-0.25,0.25],
    [0.25,-0.25,0.25],
    [-0.25,0.25,0.25],
    [0.25,0.25,0.25]
])
thruster_positions = np.array([
    [0,0,0],
    [5,0,0],
    [0,5,0],
    [5,5,0]
])
position_offSet = np.array(
    [0,0,3]
)
payload_mass = 8  # kg
payload_size = 0.5  # meters
robot_size = [5,5,3]  # meters
max_motor_torque_oz_in = 44  # oz-in
motor_rpm = 2500
drum_radius_in = 3
oz_inches_to_newton_meters = 0.00706155
max_motor_torque = max_motor_torque_oz_in * oz_inches_to_newton_meters  # Newton meters
gear_ratio = 20
spool_length_in = 6  # inches
inch_to_meters = 0.0254
drum_radius_m = drum_radius_in *inch_to_meters# meters
spool_length_m = spool_length_in * inch_to_meters  # meters
cable_diameter_in = 3/32  # inches
cable_diameter_m = cable_diameter_in * inch_to_meters  # meters

cdpr = CDPR(stands, attachment_points, payload_mass, payload_size, robot_size,
            max_motor_torque_oz_in, gear_ratio, motor_rpm, drum_radius_m,spool_length_m,cable_diameter_m)
cdpr.determineCharacteristics()
# Plot the CDPR
cdpr.plot(np.array([2.5, 2.5, 2.5]))

# Find and plot the workspace
acceleration = np.array([0,0,0])
cdpr.find_workspace(acceleration)
def generateTraj(start,end):
    return np.linspace(start,end,100)
    