import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm

# from test_camera import test_camera
from launch_noSDS_camera import get_camera_poses

def plot_camera_pose(T, ax=None, label='Camera', color='r', our_dataset=False, vert_ax='y', hori_ax='x'):
    """
    view_camera
    T: 4x4 pose c2w
    """
    if ax is None:
        fig, ax = plt.subplots()

    position = T[:3, 3]
    x, y, z = position

    if our_dataset:
        x *= 15
        y *= 15
        z *= 15

    direction = -T[:3, 2]  # forward_vector
    dir_x, dir_y, dir_z = direction


    # normalize
    arrow_length = 0.5
    norm = np.linalg.norm(direction)
    if norm == 0:
        print('camera vector length is 0')
        normalized_direction = direction
    else:
        normalized_direction = direction / norm

    # fix length
    dir_x, dir_y, dir_z = normalized_direction[0], normalized_direction[1], normalized_direction[2]
    arrow_dx = arrow_length * dir_x
    arrow_dy = arrow_length * dir_y
    arrow_dz = arrow_length * dir_z

    if (vert_ax == 'y') & (hori_ax == 'x'):
        # ax.plot(x, y, 'o', color=(139/255, 200/255, 227/255))
        ax.plot(x, y, 'o', color=color)

        ax.arrow(x, y, arrow_dx, arrow_dy, head_width=0.1, head_length=0.1, fc=color, ec=color)
        print("arrow: x={:.2f}, y={:.2f}, z={:.2f}, dx={:.2f}, dy={:.2f}, dz={:.2f}".format(x, y, z, dir_x, dir_y, dir_z))
        
        ax.text(x, y, f' {label}', fontsize=12, verticalalignment='bottom')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('camera_pose_top_2_down')
        ax.set_aspect('equal')
        ax.grid(True)
    
    elif (vert_ax == 'z') & (hori_ax == 'x'):
        ax.plot(x, z, 'o', color=color)

        ax.arrow(x, z, arrow_dx, arrow_dz, head_width=0.1, head_length=0.1, fc=color, ec=color)
        print("arrow: x={:.2f}, y={:.2f}, z={:.2f}, dx={:.2f}, dy={:.2f}, dz={:.2f}".format(x, y, z, dir_x, dir_y, dir_z))
        
        ax.text(x, z, f' {label}', fontsize=12, verticalalignment='bottom')

        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_title('camera_pose_z-x')
        ax.set_aspect('equal')
        ax.grid(True)

    elif (vert_ax == 'z') & (hori_ax == 'y'):
        ax.plot(y, z, 'o', color=color)

        ax.arrow(y, z, arrow_dy, arrow_dz, head_width=0.1, head_length=0.1, fc=color, ec=color)
        print("arrow: x={:.2f}, y={:.2f}, z={:.2f}, dx={:.2f}, dy={:.2f}, dz={:.2f}".format(x, y, z, dir_x, dir_y, dir_z))
        
        ax.text(y, z, f' {label}', fontsize=12, verticalalignment='bottom')

        ax.set_xlabel('Y')
        ax.set_ylabel('Z')
        ax.set_title('camera_pose_z-y')
        ax.set_aspect('equal')
        ax.grid(True)

    else:
        raise NotImplementedError


if __name__ == '__main__':

    # SET THIS
    save_path = "data/view_camera/18_new_cv_y-x.png"
    vert_ax = 'y'
    hori_ax = 'x'
    view_space = 1
    cam_file = 'data/cameras.npz'
    _, poses, _ = get_camera_poses(cam_file) 
    # poses = test_camera()
    our_dataset = True



    # colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    num_cameras = len(poses)
    cmap = cm.get_cmap('Spectral')  # colormap
    colors = [cmap(i / num_cameras) for i in range(num_cameras)] 
        
    fig, ax = plt.subplots(figsize=(10, 8))

    for i, T in enumerate(poses):
        if i % view_space == 0:
            label = f'C{i}'
            color = colors[i % len(colors)]  
            plot_camera_pose(T, ax=ax, label=label, color=color, our_dataset=our_dataset, vert_ax=vert_ax, hori_ax=hori_ax)

    # plt.show()
    plt.savefig(save_path)
