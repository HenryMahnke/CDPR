import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties

def text_to_splines(text, num_points=100, font_size=50):
    font_prop = FontProperties(family='sans-serif', style='normal')
    
    all_splines = []
    x_offset = 0
    
    for char in text:
        try:
            path = TextPath((x_offset, 0), char, prop=font_prop, size=font_size)
        except TypeError:
            # Fallback if 'prop' is not accepted
            path = TextPath((x_offset, 0), char, size=font_size)
        
        for verts, codes in path.iter_segments():
            if len(verts) > 2:  # This is a curve
                x, y = verts.reshape(-1, 2).T
                
                # Create a parametric variable
                t = np.linspace(0, 1, len(x))
                
                # Create interpolation functions
                fx = interpolate.interp1d(t, x, kind='cubic')
                fy = interpolate.interp1d(t, y, kind='cubic')
                
                # Generate points along the spline
                t_new = np.linspace(0, 1, num_points)
                x_new = fx(t_new)
                y_new = fy(t_new)
                
                all_splines.append(np.column_stack((x_new, y_new)))
        
        x_offset += path.get_extents().width
    
    return all_splines

def plot_splines(splines):
    fig, ax = plt.subplots(figsize=(10, 5))
    
    for spline in splines:
        ax.plot(spline[:, 0], spline[:, 1], 'b-')
    
    ax.set_aspect('equal', 'box')
    ax.invert_yaxis()
    plt.title("Text as Splines")
    plt.axis('off')
    plt.show()

# Usage
text = "Hello"
splines = text_to_splines(text, num_points=100, font_size=50)
plot_splines(splines)

# Print spline data for robot
for i, spline in enumerate(splines):
    print(f"Spline {i + 1}:")
    for point in spline:
        print(f"  X: {point[0]:.2f}, Y: {point[1]:.2f}")