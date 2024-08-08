import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

class SplineTrajectoryGenerator3D:
    def __init__(self, vel_limit=1, acc_limit=1):
        self.points = []
        self.spline = None
        self.total_time = None
        self.vel_limit = vel_limit
        self.acc_limit = acc_limit
        self.time_scaling_factor =1

    def add_point(self, x, y, z):
        self.points.append((x, y, z))

    def generate_spline(self):
        if len(self.points) < 2:
            raise ValueError("At least two points are required to generate a spline.")
        
        x, y, z = zip(*self.points)
        t = np.linspace(0, 1, len(self.points))
        
        self.spline = interpolate.CubicSpline(t, list(zip(x, y, z)), bc_type='natural')
        self.total_time = 1  # Normalized time
        self.apply_limits()
        
    def apply_limits(self):
        t = np.linspace(0,1, 1000)
        positions = self.spline(t) #getting the positions
        arc_length = np.sum(np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))) #calculating the arc length
        
        if self.vel_limit is not None:
            self.total_time = max(self.total_time, arc_length / self.vel_limit)
        velocities = self.spline(t, 1) / self.total_time
        accelerations = self.spline(t, 2) / (self.total_time ** 2)
        vel_magnitudes = np.linalg.norm(velocities, axis=1)
        acc_magnitudes = np.linalg.norm(accelerations, axis=1)
        
        time_scale_vel = 1
        time_scale_acc = 1
        
        if self.vel_limit is not None:
            max_vel = np.max(vel_magnitudes)
            if max_vel > self.vel_limit:
                time_scale_vel = self.vel_limit / max_vel
        if self.acc_limit is not None:
            max_acc = np.max(acc_magnitudes)
            if max_acc > self.acc_limit:
                time_scale_acc = np.sqrt(self.acc_limit / max_acc)
        self.time_scaling_factor = max(time_scale_vel, time_scale_acc)
        self.total_time *= self.time_scaling_factor

    def get_position(self, time):
        if self.spline is None:
            raise ValueError("Spline has not been generated yet.")
        
        if time < 0 or time > self.total_time:
            raise ValueError("Time must be between 0 and total time.")
        
        normalized_time = time / self.total_time
        return self.spline(normalized_time)

    def get_velocity(self, time):
        if self.spline is None:
            raise ValueError("Spline has not been generated yet.")
        
        if time < 0 or time > self.total_time:
            raise ValueError("Time must be between 0 and total time.")
        
        normalized_time = time / self.total_time
        return self.spline(normalized_time, 1) / self.total_time

    def get_acceleration(self, time):
        if self.spline is None:
            raise ValueError("Spline has not been generated yet.")
        
        if time < 0 or time > self.total_time:
            raise ValueError("Time must be between 0 and total time.")
        
        normalized_time = time / self.total_time
        return self.spline(normalized_time, 2) / (self.total_time ** 2)

    def plot_trajectory(self):
        if self.spline is None:
            raise ValueError("Spline has not been generated yet.")
        
        t = np.linspace(0, 1, 500)
        positions = self.spline(t)
        velocities = self.spline(t, 1) / self.total_time
        accelerations = self.spline(t, 2) / (self.total_time ** 2)
        
        vel_magnitude = np.linalg.norm(velocities, axis=1)
        acc_magnitude = np.linalg.norm(accelerations, axis=1)

        fig, axs = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('Velocity and Acceleration Profiles', fontsize=16)

        # Velocity plots
        axs[0, 0].plot(t, velocities[:, 0], 'r-')
        axs[0, 0].set_title('X Velocity')
        axs[0, 0].set_xlabel('Time')
        axs[0, 0].set_ylabel('Velocity (m/s)')

        axs[0, 1].plot(t, velocities[:, 1], 'g-')
        axs[0, 1].set_title('Y Velocity')
        axs[0, 1].set_xlabel('Time')
        axs[0, 1].set_ylabel('Velocity (m/s)')

        axs[0, 2].plot(t, velocities[:, 2], 'b-')
        axs[0, 2].set_title('Z Velocity')
        axs[0, 2].set_xlabel('Time')
        axs[0, 2].set_ylabel('Velocity (m/s)')

        axs[0, 3].plot(t, vel_magnitude, 'k-')
        axs[0, 3].set_title('Velocity Magnitude')
        axs[0, 3].set_xlabel('Time')
        axs[0, 3].set_ylabel('Velocity (m/s)')

        # Acceleration plots
        axs[1, 0].plot(t, accelerations[:, 0], 'r-')
        axs[1, 0].set_title('X Acceleration')
        axs[1, 0].set_xlabel('Time')
        axs[1, 0].set_ylabel('Acceleration (m/s²)')

        axs[1, 1].plot(t, accelerations[:, 1], 'g-')
        axs[1, 1].set_title('Y Acceleration')
        axs[1, 1].set_xlabel('Time')
        axs[1, 1].set_ylabel('Acceleration (m/s²)')

        axs[1, 2].plot(t, accelerations[:, 2], 'b-')
        axs[1, 2].set_title('Z Acceleration')
        axs[1, 2].set_xlabel('Time')
        axs[1, 2].set_ylabel('Acceleration (m/s²)')

        axs[1, 3].plot(t, acc_magnitude, 'k-')
        axs[1, 3].set_title('Acceleration Magnitude')
        axs[1, 3].set_xlabel('Time')
        axs[1, 3].set_ylabel('Acceleration (m/s²)')

        plt.tight_layout()
        plt.show()

        # 3D trajectory plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', label='Trajectory')
        ax.scatter(*zip(*self.points), color='red', s=50, label='Control Points')
        ax.set_title('3D Trajectory')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.show()

# Example usage
generator = SplineTrajectoryGenerator3D(0.5,0.5)

# Add 3D points
generator.add_point(0, 0, 0)
generator.add_point(1, 2, 3)
generator.add_point(3, 1, 2)
generator.add_point(4, 4, 1)
generator.add_point(5, 3, 5)

# Generate spline
generator.generate_spline()

# Plot the trajectory, velocities, and accelerations
generator.plot_trajectory()
print(generator.total_time)