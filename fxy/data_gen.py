import numpy as np

def f(x, y):
    return np.exp(np.sin(np.pi*x)+y**2)

def generate_train_data(size, func=f, noise=0., seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, size)
    y = rng.uniform(0, 1, size)
    fv = func(x, y)
    noise_vals = rng.normal(loc=0.0, scale=noise, size=size)
    fv_noisy = fv + noise_vals
    return np.column_stack((x, y, fv_noisy))

def generate_test_data(size, func=f):
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
    seed = 0
    size = 1_000_000  # number of points per axis
    data = generate_train_data(size, noise=0.5, seed=0)
    # data = generate_train_data(size)
    
    # Save to CSV
    output_path = 'train_data.csv'
    np.savetxt(output_path, data, delimiter=',')
    print(f"Data saved to {output_path} (shape: {data.shape})")
    # data = np.loadtxt("test_data.csv", delimiter=",")