import pytest 
import numpy as np
from xempi import xeMPI



@pytest.fixture
def xempi_instance():
    return xeMPI() #init xeMPI once per test session


def test_matvec(xempi_instance): 

    A = np.array([[1.0, 2.0, 3.0, 4.0],
                       [5.0, 6.0, 7.0, 8.0],
                       [9.0, 10.0, 11.0, 12.0]], dtype=np.float32)
    v = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    expected = np.dot(A, v)
    
    A_gpu = xempi_instance.send_to_gpu(A.flatten())
    v_gpu = xempi_instance.send_to_gpu(v)
    result_buf = xempi_instance.execute_matvec(A_gpu, v_gpu, 3, 4)
    result = xempi_instance.receive_from_gpu(result_buf, (3,))
    print(f'xeMPI result: {result}')
    print(f'Expected result: {expected}')

    assert np.allclose(result, expected, rtol=1e-5), "Matvec test failed"

def test_dot(xempi_instance): 

    size = 1000
    v1 = np.random.rand(size).astype(np.float32)
    v2 = np.random.rand(size).astype(np.float32)
    expected = np.dot(v1, v2)
    
    v1_gpu = xempi_instance.send_to_gpu(v1)
    v2_gpu = xempi_instance.send_to_gpu(v2)
    result = xempi_instance.execute_dot_product(v1_gpu, v2_gpu, size)
    print(f'xeMPI result: {result}')
    print(f'Expected result: {expected}')

    assert np.allclose(result, expected, rtol=1e-5), "Dot product test failed"
