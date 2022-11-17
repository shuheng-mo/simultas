from halo_exchange_upgraded import HaloExchange
from time import perf_counter
import numpy as np

start = perf_counter()
h = HaloExchange(structured=True,tensor_used=False,double_precision=True,corner_exchanged=True)
mesh = np.random.random((4,4))
print(mesh)
h.initialization(mesh,is_periodic=False,is_reordered=False)
# dim = HaloExchange.generate_proc_dim_3D(8)
# dim = HaloExchange.generate_proc_dim_2D(16)
# print(dim)
# h.clear_cache()

end = perf_counter()
print(end - start)

arr = np.array([1,2,3,4,5])
d = arr[1:3] # slice throws a new copy
d[1] = 4
arr[1:3] = d
print(arr)


