from xempi import xeMPI
import numpy as np


mpi = xeMPI()

print("Using GPU:", mpi.gpu.name)


# Test code to append to mpi.py (replace the existing if __name__ block)
if __name__ == "__main__":
    import time

    def run_matvec_test(mpi, matrix, vector, rows, cols, test_name):
        """Run a single matvec test with timing and validation."""
        print(f"\n--- {test_name} ---")
        print(f"Matrix size: {rows}x{cols}, Vector size: {cols}")
        
        # CPU computation for reference
        start_cpu = time.time()
        expected = np.dot(matrix, vector)
        cpu_time = time.time() - start_cpu
        
        # GPU computation
        start_gpu = time.time()
        matrix_gpu = mpi.send_to_gpu(matrix.flatten())
        vector_gpu = mpi.send_to_gpu(vector)
        result_buf = mpi.gpu_execute_matvec(matrix_gpu, vector_gpu, rows, cols)
        result = mpi.receive_from_gpu(result_buf, (rows,))
        gpu_time = time.time() - start_gpu
        
        # Validation
        print("GPU Result:", result)
        print("Expected (CPU):", expected)
        print(f"CPU Time: {cpu_time:.6f} seconds")
        print(f"GPU Time: {gpu_time:.6f} seconds")
        print("Match:", np.allclose(result, expected, rtol=1e-5, atol=1e-5))

    # Initialize MPI

    print("Using GPU:", mpi.gpu.name)

    # Test 1: Small matrix (3x4)
    matrix_small = np.array([[1.0, 2.0, 3.0, 4.0],
                             [5.0, 6.0, 7.0, 8.0],
                             [9.0, 10.0, 11.0, 12.0]], dtype=np.float32)
    vector_small = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    run_matvec_test(mpi, matrix_small, vector_small, 3, 4, "Small Matrix Test")

    # Test 2: Medium matrix (100x100)
    matrix_med = np.random.rand(100, 100).astype(np.float32)
    vector_med = np.random.rand(100).astype(np.float32)
    run_matvec_test(mpi, matrix_med, vector_med, 100, 100, "Medium Matrix Test")

    # Test 3: Large matrix (1000x1000)
    matrix_large = np.random.rand(20000, 20000).astype(np.float32)
    vector_large = np.random.rand(20000).astype(np.float32)
    run_matvec_test(mpi, matrix_large, vector_large, 20000, 20000, "Large Matrix Test")

    # Test 4: Edge case - Single element (1x1)
    matrix_single = np.array([[2.0]], dtype=np.float32)
    vector_single = np.array([3.0], dtype=np.float32)
    run_matvec_test(mpi, matrix_single, vector_single, 1, 1, "Single Element Test")

    # Test 5: Edge case - Tall and thin (10x1)
    matrix_tall = np.ones((10, 1), dtype=np.float32)
    vector_tall = np.array([2.0], dtype=np.float32)
    run_matvec_test(mpi, matrix_tall, vector_tall, 10, 1, "Tall and Thin Test")

    # Test 6: Edge case - Wide and short (1x10)
    matrix_wide = np.ones((1, 10), dtype=np.float32)
    vector_wide = np.ones(10, dtype=np.float32)
    run_matvec_test(mpi, matrix_wide, vector_wide, 1, 10, "Wide and Short Test")