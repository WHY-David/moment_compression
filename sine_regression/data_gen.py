import numpy as np

def f(x, y):
    """Compute f(x, y) = exp(sin(pi * x) + y^2)."""
    return np.exp(np.sin(np.pi * x) + y**2)

def generate_data(size, func):
    """
    Generate grid data of shape (size^2, 3) containing columns (x, y, f(x, y)).
    
    Parameters:
    - size (int): Number of points along each axis.
    - func (callable): Function that takes two arrays (x, y) and returns f(x, y).
    
    Returns:
    - np.ndarray: Array of shape (size^2, 3).
    """
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    xv, yv = np.meshgrid(x, y, indexing='xy')
    xv_flat = xv.ravel()
    yv_flat = yv.ravel()
    fv_flat = func(xv_flat, yv_flat)
    return np.column_stack((xv_flat, yv_flat, fv_flat))

if __name__ == "__main__":
    # Example usage
    size = 101  # number of points per axis
    data = generate_data(size, f)
    
    # Save to CSV
    output_path = '/mnt/data/data.csv'
    np.savetxt(output_path, data, delimiter=',', header='x,y,f', comments='')
    print(f"Data saved to {output_path} (shape: {data.shape})")