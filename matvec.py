from mpi import MPI
import numpy as np


mpi = MPI()

print("Using GPU:", mpi.gpu.name)



A = np.array([[1.0, 2.0, 3.0, 4.0],
                       [5.0, 6.0, 7.0, 8.0],
                       [9.0, 10.0, 11.0, 12.0]], dtype=np.float32)


v = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)


A_gpu = mpi.send_to_gpu(A.flatten())
v_gpu = mpi.send_to_gpu(v)

result = mpi.gpu_execute_matvec(A_gpu, v_gpu, rows=3, cols=4)
result = mpi.receive_from_gpu(result, (3,))

print("Matvec A.v = ", result)
print("Expected result", np.dot(A, v))