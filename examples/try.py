from halo_exchange_upgraded import HaloExchange
from time import perf_counter
from pprint import pprint
import numpy as np

start = perf_counter()
h = HaloExchange(structured=True, tensor_used=False,
                 double_precision=True, corner_exchanged=True)

mesh = np.array([[0.78580622, 0.60408768, 0.47037418, 0.46169217],
                 [0.06124935, 0.01102084, 0.12187897, 0.66882098],
                 [0.05242322, 0.65417555, 0.53059087, 0.18224255],
                 [0.4206126, 0.27398225, 0.24499399, 0.8372037]])
# pprint(mesh)
x, y, sub_domain = h.initialization(
    mesh, is_periodic=False, is_reordered=False)

# print(f'{x},{y}')
# pprint(sub_domain)

h.clear_cache()
end = perf_counter()
print(end - start)
