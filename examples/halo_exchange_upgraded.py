# imports
import sys
import numpy as np  # cupy can be used as optimisation if CUDA/AMD GPUs are available
from tensorflow import keras
import tensorflow as tf
from mpi4py import MPI
from pprint import pprint

# default global settings
assert tf.__version__ >= "2.0"
# np.set_printoptions(threshold=sys.maxsize)  # print out the full numpy array
np.set_printoptions(suppress=True)

# Halo Exchange class


class HaloExchange:
    def __init__(self, structured=True, tensor_used=False, double_precision=False, corner_exchanged=False) -> None:
        self.comm = None
        self.rank = 0
        self.num_process = 1
        self.rows, self.cols = 1, 1
        self.sub_nx, self.sub_ny, self.sub_nz = 0, 0, 0
        self.neighbors = [-2] * 26  # suppose we support up to 3D, 26 neighbors
        self.current_vector = None
        self.current_matrix = None
        self.current_cuboid = None
        self.structured_mesh = structured
        self.is_tensor_mesh = tensor_used
        self.is_double_precision = double_precision
        self.is_corner_exchanged = corner_exchanged

    # member functions
    def mpi_init(self, proc_grid_dim, is_periodic, is_reordered):
        """Initialize the MPI topology and other MPI processes

        Args:
            proc_grid_dim (tuple): 2D or 3D cartesian grid coordinate in format (x,y) or (z,x,y)
            is_periodic (bool): flag to control periodic boundary condition
            is_reordered (bool): flag to control the cartesian topology is reordered

        Returns:
            None
        """
        # self.num_process = MPI.COMM_WORLD.Get_size()  # get number of process
        # assert self.num_process > 1, f"Parallelisation involves 2 or more processes, otherwise run implementation without MPI"

        # create Cartesian Topology
        self.comm = MPI.COMM_WORLD.Create_cart(
            proc_grid_dim,
            periods=is_periodic,  # set True if periodic boundary
            reorder=is_reordered)

        # get the rank of current process
        self.rank = self.comm.rank

    def initialization(self, mesh, is_periodic=False, is_reordered=False):
        # convert to numpy array if it is tensor
        if tf.is_tensor(mesh):
            mesh = mesh.numpy()

        # remove the extra one dimensions
        mesh = HaloExchange.remove_one_dims(mesh)

        num_process = MPI.COMM_WORLD.Get_size()
        proc_grid_dim = (1, 1, 1)
        if mesh.ndim == 2:
            proc_grid_dim = HaloExchange.generate_proc_dim_2D(num_process)
        elif mesh.ndim == 3:
            proc_grid_dim = HaloExchange.generate_proc_dim_3D(num_process)

        assert num_process > 1, f'Come on, one process means serial!!!'
        self.mpi_init(proc_grid_dim, is_periodic,
                      is_reordered)  # mpi initialization

        # if mesh.ndim == 1 or mesh.ndim == 2:
        #     if mesh.ndim == 1:
        #         domain_decomposition_strip()
        #         # domain_decomposition_1D() # is this really just strip decomposition
        #     elif mesh.ndim == 2 and (proc_grid_dim[0]==1 or proc_grid_dim[1]==1):
        #         domain_decomposition_strip()
        #     else:
        #         domain_decomposition_2D()
        # elif mesh.ndim == 3:
        #     domain_decomposition_3D()

    def clear_cache(self):
        """clear all cached problem domains

        Returns:
            None
        """
        if self.current_vector != None:
            del self.current_vector
        elif self.current_matrix != None:
            del self.current_matrix
        elif self.current_cuboid != None:
            del self.current_cuboid

    # static methods

    @staticmethod
    def generate_proc_dim_2D(num_process):
        """generate 2D cartesian grid coordinate by number of processors

        Args:
            num_process (int): number of processors

        Returns:
            int,int: X,Y
        """
        rows, cols = 0, 0
        min_gap = num_process
        max_val = int(num_process**0.5 + 1)
        for i in range(1, max_val+1):
            if num_process % i == 0:
                gap = abs(num_process/i - i)
                if gap < min_gap:
                    min_gap = gap
                    rows = i
                    cols = int(num_process / i)

        return (rows, cols)

    @staticmethod
    def generate_proc_dim_3D(num_process):
        """

        Args:
            num_process (_type_): _description_

        Returns:
            _type_: _description_
        """
        assert num_process >= 1, f'The number of processors should be greater or equal to 1'
        left = 1
        right = num_process
        while right - left > 1e-5:
            mid = left + (right - left)//2
            cube_val = mid ** 3
            if cube_val == num_process:
                return (int(mid), int(mid), int(mid))
            elif cube_val > num_process:
                right = mid - 0.01
            elif cube_val < num_process:
                left = mid + 0.01

        if left**3 != num_process:
            return (-1, -1, -1)

        return (left, left, left)

    @staticmethod
    def remove_one_dims(mesh):
        """remove the trivial 1-dimensions from the tensor object

        Args:
            input (numpy array): numpy array that converted from the tensor

        Returns:
            np.array: the squeezed numpy array
        """
        while mesh.ndim >= 3:
            if mesh.shape[0] == 1 and mesh.shape[-1] == 1:
                mesh = np.squeeze(mesh, axis=0)
                mesh = np.squeeze(mesh, axis=-1)
        return mesh


############################## Util functions ##################################

def id_2_idx(rank, cols):
    """convert process rank to cartesian grid coordiantes

    Args:
        rank (int): process rank
        cols (int): length of x coordinates of grid coordiantes

    Returns:
        int,int: 2D cartesian grid coordinate (x,y)
    """
    return rank/cols, rank % cols


def idx_2_id(rows, cols, id_row, id_col):
    """convert 2D cartesian grid coordinates to process rank

    Args:
        rows (int): Y
        cols (int): X
        id_row (int): y
        id_col (int): x

    Returns:
        int: process rank
    """
    if id_row >= rows or id_row < 0:
        return -1
    if id_col >= cols or id_col < 0:
        return -1
    return id_row * id_col + id_col


def domain_decomposition_strip(mesh, num_process):
    """strip decomposition for 2D block structured mesh

    Args:
        mesh (numpy array): problem mesh
        num_process (int): number of processors

    Returns:
        list of numpy: divided sub-domains 
    """
    sub_domains = np.hsplit(mesh, num_process)  # split the domain horizontally
    return sub_domains


def domain_decomposition_grid(mesh, rows, cols):
    """grid decomposition for 2D block-structured meshes

    Args:
        mesh (numpy array): problem mesh
        rows (int): X
        cols (int): Y

    Returns:
        list of numpy: sub-domains
    """
    nx, ny = mesh.shape
    assert nx % rows == 0, f"{nx} rows is not evenly divisible by {rows}"
    assert ny % cols == 0, f"{ny} cols is not evenly divisible by {cols}"
    sub_nx = nx//rows
    sub_ny = ny//cols
    return (mesh.reshape(nx//sub_nx, sub_nx, -1, sub_ny)
            .swapaxes(1, 2)
            .reshape(-1, sub_nx, sub_ny))


def domain_decomposition_cube(mesh, proc_grid_dim):
    """grid decomposition for 3D block structured mesh

    Args:
        mesh (numpy array): problem mesh
        proc_grid_dim (tuple): (Z,X,Y)

    Returns:
        list of numpy: sub-domains
    """
    global sub_nx, sub_ny, sub_nz
    nx, ny, nz = mesh.shape

    assert nx % proc_grid_dim[0] == 0, f"{nx} grids along x axis is not evenly divisible by {proc_grid_dim[0]}"
    assert ny % proc_grid_dim[1] == 0, f"{ny} grids along y axis is not evenly divisible by {proc_grid_dim[1]}"
    assert nz % proc_grid_dim[2] == 0, f"{nz} grids along z axis is not evenly divisible by {proc_grid_dim[2]}"

    sub_nx = nx // proc_grid_dim[0]
    sub_ny = ny // proc_grid_dim[1]
    sub_nz = nz // proc_grid_dim[2]

    new_shape = (sub_nx, sub_ny, sub_nz)
    num_cubes = np.array(mesh.shape) // new_shape
    split_shape = np.column_stack([num_cubes, new_shape]).reshape(-1)
    order = np.array([0, 2, 4, 1, 3, 5])

    # return a numpy array
    return mesh.reshape(split_shape).transpose(order).reshape(-1, *new_shape)


def padding_block_halo_1D(sub_domain, halo_size, halo_val=0):
    """padding the 1D subdomain with halos manually

    Args:
        sub_domain (numpy array): 1D sub-domain
        halo_size (int): width of the halo grids
        halo_val (int, optional): values to fill into the halo grids. Defaults to 0.

    Returns:
        numpy array: sub-domain with paddings
    """
    if tf.is_tensor(sub_domain):
        sub_domain = sub_domain.numpy()

    if sub_domain.ndim > 1:
        sub_domain = np.squeeze(sub_domain, axis=0)
        sub_domain = np.squeeze(sub_domain, axis=-1)

    return np.pad(sub_domain, (halo_size, halo_size), 'constant', constant_values=(halo_val,))


def padding_block_halo_2D(sub_domain, halo_size, halo_val=0):
    """padding the 2D subdomain with halos manually

    Args:
        sub_domain (numpy array): 2D sub-domain
        halo_size (int): width of the halo grids
        halo_val (int, optional): values to fill into the halo grids. Defaults to 0.

    Returns:
        numpy array: sub-domains with paddings
    """
    if tf.is_tensor(sub_domain):
        sub_domain = sub_domain.numpy()

    if sub_domain.ndim > 2:
        sub_domain = np.squeeze(sub_domain)

    # note padding halo values with 0 by default
    return np.pad(sub_domain, (halo_size, halo_size), 'constant', constant_values=(halo_val,))


def padding_block_halo_2D_custom(sub_domain, halo_dim, halo_vals):
    """_summary_

    Args:
        sub_domain (_type_): _description_
        halo_dim (_type_): _description_
        halo_vals (_type_): _description_

    Returns:
        _type_: _description_
    """
    pass


def padding_block_halo_3D(sub_cube, halo_size, halo_val=0):
    """padding the 3D subdomain with halos manually

    Args:
        sub_cube (numpy array): 3D sub-domain
        halo_size (int): width of the halo grids
        halo_val (int, optional): values to fill into the halos. Defaults to 0.

    Returns:
        _type_: _description_
    """
    if tf.is_tensor(sub_cube):
        sub_cube = sub_cube.numpy()

    if sub_cube.ndim > 3:
        sub_cube = remove_one_dims(sub_cube)

    nx, ny, nz = sub_cube.shape

    # note padding halo values with 0 by default
    ans = np.pad(sub_cube, (halo_size, halo_size),
                 'constant', constant_values=(halo_val,))
    return tf.convert_to_tensor(ans.reshape(1, nx+2, ny+2, nz+2, 1))


def padding_block_halo_3D_custom(sub_cube, halo_dim, halo_vals):
    """_summary_

    Args:
        sub_cube (_type_): _description_
        halo_dim (_type_): _description_
        halo_vals (_type_): _description_

    Returns:
        _type_: _description_
    """
    pass


############################## MPI Initialization ##############################
def domain_decomposition_1D(values, nx, is_periodic=False, is_reordered=True):
    """domain decomposition for 1D block structured meshes

    Args:
        values (numpy array): the input problem mesh
        nx (int): shape of the input problem mesh (nx,)
        is_periodic (bool, optional): flag to control the periodic boundary condition. Defaults to False.
        is_reordered (bool, optional): flag to control the topology is reordered. Defaults to True.

    Returns:
        int, numpy array: single sub-domain shape, sub-domains
    """
    global num_process, comm, rank, sub_nx, neighbors, current_vector
    LEFT = 0
    RIGHT = 1

    num_process = MPI.COMM_WORLD.Get_size()  # get number of process
    proc_grid_dim = (num_process,)
    mpi_initialization(proc_grid_dim, is_periodic, is_reordered)

    neighbors[LEFT], neighbors[RIGHT] = comm.Shift(0, 1)

    # print("NEIGHBORS OF {} ".format(rank),neighbors)

    sub_domains = domain_decomposition_strip(
        values.reshape(nx,), num_process)  # (1,x)
    sub_nx = sub_domains[rank].shape[0]
    current_vector = np.pad(
        sub_domains[rank], (1, 1), "constant", constant_values=(0,))

    return sub_nx, current_vector


def domain_decomposition_2D(values, nx, ny, is_periodic=(False, False), is_reordered=True):
    """domain decomposition for 2D block structured meshes

    Args:
        values (numpy array): problem mesh
        nx (int): x shape of the problem mesh
        ny (int): y shape of the problem mesh
        is_periodic (tuple, optional): flag to control the periodic boundary. Defaults to (False, False).
        is_reordered (bool, optional): flag to control the topology is reordered. Defaults to True.

    Returns:
        int,int,numpy array: sub-domain shape x, sub-domain shape y, sub-domains
    """
    global comm, rank, sub_nx, sub_ny, neighbors, current_domain

    TOP = 0
    BOTTOM = 1
    LEFT = 2
    RIGHT = 3

    num_process = MPI.COMM_WORLD.Get_size()  # get number of process
    proc_grid_dim = generate_dimension_2D(num_process)
    mpi_initialization(proc_grid_dim, is_periodic, is_reordered)

    if rows == 1 or cols == 1:
        sub_domains = domain_decomposition_strip(
            values.reshape(nx, ny), num_process)  # 2 process
        sub_nx, sub_ny = sub_domains[0].shape
    else:
        # if the process arrays is 2D then use grid decomposition to split domain
        sub_domains = domain_decomposition_grid(
            values.reshape(nx, ny), rows, cols)
        sub_nx, sub_ny = sub_domains[0].shape

    # create customized MPI datatype
    # grid_size = [sub_nx, sub_ny]
    # subdomain_size = [sub_nx+2, sub_ny + 2]
    # start_indices = [1, 1]

    # customised data type for 2D problem
    # comm_datatype = MPI.DOUBLE.Create_subarray(subdomain_size, grid_size, start_indices).Commit()

    # find the processor id of all neighboring processors
    neighbors[TOP], neighbors[BOTTOM] = comm.Shift(0, 1)
    neighbors[LEFT],  neighbors[RIGHT] = comm.Shift(1, 1)

    current_domain = np.pad(
        sub_domains[rank], (1, 1), "constant", constant_values=(0,))

    return sub_nx, sub_ny, current_domain


def domain_decomposition_3D(values, nx, ny, nz, is_periodic=(False, False, False), is_reordered=True):
    """domain decomposition for 3D block structured meshes

    Args:
        values (numpy array): problem mesh
        nx (int): x shape of the problem mesh
        ny (int): y shape of the problem mesh
        nz (int): z shape of the problem mesh
        is_periodic (tuple, optional): flags to control the periodic boundary. Defaults to (False, False, False).
        is_reordered (bool, optional): flags to control the topology is reordered. Defaults to True.

    Returns:
        int,int,int,numpy array: sub-domain shape x, sub-domain shape y, sub-domain shape z, sub-domains
    """
    global rank, num_process, neighbors, current_cube

    # neighbor indices
    LEFT = 0
    RIGHT = 1
    FRONT = 2
    BEHIND = 3
    TOP = 4
    BOTTOM = 5

    num_process = MPI.COMM_WORLD.Get_size()  # get number of process
    # TODO: create Cartesian topology for processes in 3D space
    proc_grid_dim = (2, 2, 2)  # divide to 2x2x2, 8 subcubes
    # proc_grid_dim = (4,4,4) # divide to 4x4x4, 64 subcubes
    mpi_initialization(proc_grid_dim, is_periodic, is_reordered)

    # edge case, if 1 process we do nothing
    if num_process == 1:
        return nx, ny, nz, values

    if tf.is_tensor(values):
        sub_cubes = domain_decomposition_cube(tf.reshape(
            values, [nx, ny, nz]).numpy(), proc_grid_dim)
        # we can do this in complete tensorflow routines
        # current_cube = tf.convert_to_tensor(sub_cubes[rank], dtype=tf.float32)
        # paddings = tf.constant([[1, 1], [1, 1], [1, 1]])
        # current_cube = tf.pad(current_cube, paddings)
    else:
        sub_cubes = domain_decomposition_cube(values.reshape(
            nx, ny, nz), proc_grid_dim)  # if it is numpy reshape directly

    # create customized MPI datatype
    # cube_size = [sub_nx, sub_ny, sub_nz]
    # subcube_size = [sub_nx+2, sub_ny + 2, sub_nz + 2]
    # start_indices = [1, 1, 1]

    # use datatypes with 64 bits
    # comm_datatype_3D = MPI.DOUBLE.Create_subarray(subcube_size, cube_size, start_indices).Commit()

    # padding the halo grids
    current_cube = np.pad(
        sub_cubes[rank], (1, 1), 'constant', constant_values=(0,))

    # print("[CURRENT_CUBE_SHAPE of {}]".format(rank), current_cube.shape)

    # find neighbors (note here 0,1,2 are x,y,z coordinates respectively)
    neighbors[LEFT], neighbors[RIGHT] = comm.Shift(2, 1)
    neighbors[FRONT],  neighbors[BEHIND] = comm.Shift(1, 1)
    neighbors[BOTTOM],  neighbors[TOP] = comm.Shift(0, 1)

    # print("[NEIGHBORS of {}] ".format(rank),neighbors)

    # return tf.convert_to_tensor(current_cube,np.float64)
    return sub_nx, sub_ny, sub_nz, current_cube


def structured_halo_update_1D(input_vector):
    """parallel updating of halos in 1D

    Args:
        input_vector (numpy): 1D sub-domain

    Returns:
        tensor: 1D tensorflow tensor with halos updated
    """
    global comm, neighbors, current_vector, sub_nx

    if tf.is_tensor(input_vector):
        current_vector = input_vector.numpy()
    else:
        current_vector = input_vector

    if current_vector.ndim > 1:
        current_vector = np.squeeze(current_vector, axis=0)
        current_vector = np.squeeze(current_vector, axis=-1)

    #print("reduced shape: ",current_vector.shape)

    LEFT = 0
    RIGHT = 1

    sub_nx = current_vector.shape[0]

    send_left = np.copy(np.ascontiguousarray(current_vector[1]))
    recv_right = np.empty_like(send_left)
    send_right = np.copy(np.ascontiguousarray(current_vector[-2]))
    recv_left = np.empty_like(send_right)

    # Blocking communications
    # comm.Send(buf=[send_left,1,MPI.DOUBLE],dest=neighbors[LEFT],tag=11) # send to left
    # comm.Send(buf=[send_right,1,MPI.DOUBLE],dest=neighbors[RIGHT],tag=22) # send to left
    # comm.Recv(buf=[recv_right,1,MPI.DOUBLE],source=neighbors[RIGHT],tag=11)
    # comm.Recv(buf=[recv_left,1,MPI.DOUBLE],source=neighbors[LEFT],tag=22)

    # Non-blocking send-recv, which gives the same result
    requests = []
    requests.append(comm.Isend(
        [send_left, 1, MPI.DOUBLE], dest=neighbors[LEFT]))
    requests.append(comm.Isend(
        [send_right, 1, MPI.DOUBLE], dest=neighbors[RIGHT]))
    requests.append(comm.Irecv(
        [recv_right, 1, MPI.DOUBLE], source=neighbors[RIGHT]))
    requests.append(comm.Irecv(
        [recv_left, 1, MPI.DOUBLE], source=neighbors[LEFT]))
    MPI.Request.Waitall(requests)
    requests.clear()

    if neighbors[RIGHT] != -2:
        current_vector[-1] = recv_right
    if neighbors[LEFT] != -2:
        current_vector[0] = recv_left

    return tf.convert_to_tensor(current_vector.reshape(1, sub_nx, 1))


def structured_halo_update_2D(input_domain):
    """parallel updating of halos in 2D

    Args:
        input_domain (numpy): 2D sub-domain

    Returns:
        tensor: 2D tensorflow tensor with halos updated
    """
    global comm, neighbors, current_domain, sub_nx, sub_ny

    # update the values of the domain
    current_domain = input_domain

    if tf.is_tensor(input_domain):
        current_domain = input_domain.numpy()

    if current_domain.ndim > 2:
        current_domain = remove_one_dims(current_domain)

    sub_nx, sub_ny = current_domain.shape

    # neighbor indices
    TOP = 0
    BOTTOM = 1
    LEFT = 2
    RIGHT = 3

    # left and right
    # try non-blocking send and blocking receive
    send_left = np.copy(np.ascontiguousarray(current_domain[1:-1, 1]))
    recv_right = np.zeros_like(send_left)
    send_right = np.copy(np.ascontiguousarray(current_domain[1:-1, -2]))
    recv_left = np.zeros_like(send_right)

    # print("{} SEND TO LEFT THE {} :".format(rank,neighbors[LEFT]), send_left)
    # print("{} SEND TO RIGHT THE {} :".format(rank,neighbors[RIGHT]), send_right)

    # Non-blocking send-recv
    requests = []
    requests.append(comm.Isend(
        [send_left, MPI.DOUBLE_PRECISION], dest=neighbors[LEFT]))
    requests.append(comm.Isend(
        [send_right, MPI.DOUBLE_PRECISION], dest=neighbors[RIGHT]))
    requests.append(comm.Irecv(
        [recv_right, MPI.DOUBLE_PRECISION], source=neighbors[RIGHT]))
    requests.append(comm.Irecv(
        [recv_left, MPI.DOUBLE_PRECISION], source=neighbors[LEFT]))
    MPI.Request.Waitall(requests)
    requests.clear()

    # print("{} RECV FROM LEFT THE {} :".format(rank,neighbors[LEFT]), recv_left)
    # print("{} RECV FROM RIGHT THE {} :".format(rank,neighbors[RIGHT]), recv_right)

    if neighbors[RIGHT] != -2:
        current_domain[1:-1, -1] = recv_right
    if neighbors[LEFT] != -2:
        current_domain[1:-1, 0] = recv_left

    send_top = np.copy(np.ascontiguousarray(current_domain[1, :]))
    recv_bottom = np.zeros_like(send_top)
    send_bottom = np.copy(np.ascontiguousarray(current_domain[-2, :]))
    recv_top = np.zeros_like(send_bottom)

    # print("{} SEND TO TOP THE {} :".format(rank,neighbors[TOP]), send_top)
    # print("{} SEND TO BOTTOM THE {} :".format(rank,neighbors[BOTTOM]), send_bottom)

    requests = []
    requests.append(comm.Isend(
        [send_top, MPI.DOUBLE_PRECISION], dest=neighbors[TOP]))
    requests.append(comm.Isend(
        [send_bottom, MPI.DOUBLE_PRECISION], dest=neighbors[BOTTOM]))
    requests.append(comm.Irecv(
        [recv_bottom, MPI.DOUBLE_PRECISION], source=neighbors[BOTTOM]))
    requests.append(comm.Irecv(
        [recv_top, MPI.DOUBLE_PRECISION], source=neighbors[TOP]))
    MPI.Request.Waitall(requests)
    requests.clear()

    # print("{} RECV FROM TOP THE {} :".format(rank,neighbors[TOP]), recv_top)
    # print("{} RECV FROM BOTTOM THE {} :".format(rank,neighbors[BOTTOM]), recv_bottom)

    if neighbors[TOP] != -2:
        current_domain[0, :] = recv_top
    if neighbors[BOTTOM] != -2:
        current_domain[-1, :] = recv_bottom

    print("[CURRENT DOMAIN {}]".format(rank), current_domain)

    return current_domain
    # return tf.convert_to_tensor(current_domain.reshape(1, sub_nx, sub_ny, 1))


def structured_halo_update_3D(input_cube):
    """parallel updating of halos in 3D

    Args:
        input_cube (numpy): 3D sub-domain to be updated

    Returns:
        tensor: 3D tensorflow tensors with halos updated 
    """
    global current_cube, neighbors, sub_nx, sub_ny, sub_nz

    current_cube = input_cube

    if tf.is_tensor(input_cube):
        current_cube = current_cube.numpy()

    if input_cube.ndim > 3:
        current_cube = remove_one_dims(current_cube)

    sub_nx, sub_ny, sub_nz = current_cube.shape

    # neighbor indices
    LEFT = 0
    RIGHT = 1
    FRONT = 2
    BEHIND = 3
    TOP = 4
    BOTTOM = 5

    requests = []  # [ None ] * (2*nprocs) for other languages

    # FRONT AND BEHIND
    sendbuffer_1 = np.copy(np.ascontiguousarray(current_cube[1:-1, 1, 1:-1]))
    sendbuffer_2 = np.copy(np.ascontiguousarray(current_cube[1:-1, -2, 1:-1]))
    recvbuffer_1 = np.empty_like(sendbuffer_2)
    recvbuffer_2 = np.empty_like(sendbuffer_1)

    requests.append(comm.Isend(sendbuffer_1, dest=neighbors[FRONT]))
    requests.append(comm.Isend(sendbuffer_2, dest=neighbors[BEHIND]))
    requests.append(comm.Irecv(recvbuffer_1, source=neighbors[BEHIND]))
    requests.append(comm.Irecv(recvbuffer_2, source=neighbors[FRONT]))
    MPI.Request.Waitall(requests)
    requests.clear()

    # update front and behind
    if neighbors[FRONT] != -2:
        current_cube[1:-1, 0, 1:-1] = recvbuffer_2
    if neighbors[BEHIND] != -2:
        current_cube[1:-1, -1, 1:-1] = recvbuffer_1

    sendbuffer_1 = np.copy(np.ascontiguousarray(current_cube[:, :, 1]))
    sendbuffer_2 = np.copy(np.ascontiguousarray(current_cube[:, :, -2]))
    recvbuffer_1 = np.empty_like(sendbuffer_2)
    recvbuffer_2 = np.empty_like(sendbuffer_1)

    requests.append(comm.Isend(sendbuffer_1, dest=neighbors[LEFT]))
    requests.append(comm.Isend(sendbuffer_2, dest=neighbors[RIGHT]))
    requests.append(comm.Irecv(recvbuffer_1, source=neighbors[RIGHT]))
    requests.append(comm.Irecv(recvbuffer_2, source=neighbors[LEFT]))
    MPI.Request.Waitall(requests)
    requests.clear()

    if neighbors[LEFT] != -2:
        current_cube[:, :, 0] = recvbuffer_2
    if neighbors[RIGHT] != -2:
        current_cube[:, :, -1] = recvbuffer_1

    sendbuffer_1 = np.copy(np.ascontiguousarray(current_cube[-2, :, :]))
    sendbuffer_2 = np.copy(np.ascontiguousarray(current_cube[1, :, :]))
    recvbuffer_1 = np.empty_like(sendbuffer_2)
    recvbuffer_2 = np.empty_like(sendbuffer_1)

    requests.append(comm.Isend(sendbuffer_1, dest=neighbors[TOP]))
    requests.append(comm.Isend(sendbuffer_2, dest=neighbors[BOTTOM]))
    requests.append(comm.Irecv(recvbuffer_1, source=neighbors[BOTTOM]))
    requests.append(comm.Irecv(recvbuffer_2, source=neighbors[TOP]))
    MPI.Request.Waitall(requests)
    requests.clear()

    if neighbors[TOP] != -2:
        current_cube[-1, :, :] = recvbuffer_2
    if neighbors[BOTTOM] != -2:
        current_cube[0, :, :] = recvbuffer_1

    return tf.convert_to_tensor(current_cube.reshape(1, sub_nx, sub_ny, sub_nz, 1))
