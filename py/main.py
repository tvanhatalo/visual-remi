import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque

# Import necessary functions and classes from your existing code
from nnetart.network import FeedForwardNetwork
from nnetart.artgen import transform_colors

def main():
    # Parameters
    IMAGESIZE_RATIO = 2
    img_height = 1200 // IMAGESIZE_RATIO
    img_width = 1200 // IMAGESIZE_RATIO
    colormode = 'rgb'
    alpha = False  # Use alpha channel for transparency
    activation_fnc = 'sin' # sin is nice
    n_depth = 4
    n_size = 10  # Number of neurons in each layer
    symmetry = True  # Implement symmetry by squaring x and y
    trig = False  # Apply trig functions to z components alternately
    gpu = False
    total_frames = 300  # Total number of frames in the animation
    size_z = 5  # Size of the latent vector

    # New parameters for smoothing
    N = 5  # Number of past latent vectors to store
    delta_z = 0.005  # Standard deviation of the random perturbation

    # Initialize network with fixed weights
    # Modify the network to accept input of size (3 + size_z)
    my_net = FeedForwardNetwork(
        layers_dims=[n_size]*n_depth,
        activation_fnc=activation_fnc,
        colormode=colormode.lower(),
        alpha=alpha,
        input_size=3+size_z
    )

    if gpu:
        my_net = my_net.cuda()

    # Generate initial data
    initial_z = torch.zeros(size_z)
    if gpu:
        initial_z = initial_z.cuda()

    # Precompute x, y, r (they don't change over time)
    x = np.linspace(-1, 1, img_width)
    y = np.linspace(-1, 1, img_height)
    xv, yv = np.meshgrid(x, y, indexing='ij')  # Shape: (img_width, img_height)
    
    # Implement symmetry by squaring x and y if symmetry is True
    if symmetry:
        xv = xv ** 2
        yv = yv ** 2

    # Flatten x and y coordinates
    xv = xv.flatten()
    yv = yv.flatten()
    r_ = np.sqrt(xv**2 + yv**2)

    x_ = torch.from_numpy(xv).float().unsqueeze(1)
    y_ = torch.from_numpy(yv).float().unsqueeze(1)
    r_ = torch.from_numpy(r_).float().unsqueeze(1)

    num_pixels = x_.shape[0]  # Total number of pixels

    # xyr has shape (num_pixels, 3)
    xyr = torch.cat([x_, y_, r_], dim=1)
    if gpu:
        xyr = xyr.cuda()

    # Initialize z
    z = initial_z.clone()  # z is a tensor of shape (size_z,)

    # Buffers to store past N latent vectors
    past_z = deque([z.clone() for _ in range(N)], maxlen=N)

    # Prepare initial z_col based on trig setting
    if trig:
        z_col_list = []
        for i in range(size_z):
            if i % 2 == 0:
                z_col = torch.cos(z[i] * xyr[:, i % 2].squeeze())
            else:
                z_col = torch.sin(z[i] * xyr[:, i % 2].squeeze())
            z_col_list.append(z_col.unsqueeze(1))
        z_col = torch.cat(z_col_list, dim=1)
    else:
        z_col = z.unsqueeze(0).repeat(num_pixels, 1)  # Shape: (num_pixels, size_z)
        if gpu:
            z_col = z_col.cuda()

    # Concatenate xyr with z_col to form input_data
    input_data = torch.cat([xyr, z_col], dim=1)  # Shape: (num_pixels, 3 + size_z)

    # Set up matplotlib figure
    fig, ax = plt.subplots()
    plt.axis('off')

    with torch.no_grad():
        img = my_net(input_data)
    if gpu:
        img = img.cpu()

    img = img.view(img_height, img_width, -1)
    img = transform_colors(img, colormode, alpha)
    img = img.numpy()
    im = ax.imshow(img, aspect='auto', interpolation='bilinear')

    # Update function for animation
    def update(frame):
        nonlocal z, input_data, past_z

        # Compute average direction over the past N steps
        if len(past_z) >= 2:
            # Stack past z tensors into shape (N, size_z)
            past_z_tensor = torch.stack(list(past_z))  # Shape: (N, size_z)
            # Calculate differences between consecutive latent vectors
            dz = past_z_tensor[1:] - past_z_tensor[:-1]  # Shape: (N-1, size_z)
            # Compute average direction
            avg_dz = dz.mean(dim=0)  # Shape: (size_z,)
        else:
            avg_dz = torch.zeros_like(z)

        # Random perturbation
        rand_dz = torch.randn(size_z) * delta_z
        if gpu:
            rand_dz = rand_dz.cuda()

        # Update latent vectors by moving in the average direction plus random perturbation
        z = z + avg_dz + rand_dz  # Shape: (size_z,)

        # Remove the clamping of z to allow it to evolve without clipping
        # z = torch.clamp(z, -1, 1)  # This line is removed

        # Implement wrapping to keep z within [-π, π]
        z = (z + np.pi) % (2 * np.pi) - np.pi

        # Update past latent vectors
        past_z.append(z.clone())

        # Update z_col based on trig setting
        if trig:
            z_col_list = []
            for i in range(size_z):
                # Alternate between cos and sin
                if i % 2 == 0:
                    z_col = torch.cos(z[i] * xyr[:, i % 2].squeeze())
                else:
                    z_col = torch.sin(z[i] * xyr[:, i % 2].squeeze())
                z_col_list.append(z_col.unsqueeze(1))
            z_col = torch.cat(z_col_list, dim=1)
        else:
            # z_col is the same for all pixels
            z_col = z.unsqueeze(0).repeat(num_pixels, 1)  # Shape: (num_pixels, size_z)
            if gpu:
                z_col = z_col.cuda()

        # Concatenate xyr with updated z_col
        input_data = torch.cat([xyr, z_col], dim=1)
        if gpu:
            input_data = input_data.cuda()

        with torch.no_grad():
            img = my_net(input_data)
        if gpu:
            img = img.cpu()

        img = img.view(img_height, img_width, -1)
        img = transform_colors(img, colormode, alpha)
        img = img.numpy()
        im.set_data(img)
        return [im]

    # Create animation
    anim = FuncAnimation(fig, update, frames=total_frames, blit=True, interval=0.4)
    plt.show()

if __name__ == '__main__':
    main()
