import pyopencl as cl
import numpy as np

class xeMPI: 
    """
    
    matve_mul kernel: 
        get_global_id(0): gives each thread a unique ID 
    send_to_gpu:
        transfers data from CPU memory to GPU memory
        allocates a GPU memory buffer of size data.nbytes (total bytes of the array)  
    receive_from_gpu: 
        creates empty numpy array on the CPU with shape that matches the kernel output
    gpu_execute_matvec: 
        runs matvec_mul on the GPU to compute the matrix-vector product
        global_size = the total number the GPU will launch to compute the matrix-vector multiplication. How many threads the GPU to handle all rows of the matrix
    """

    def __init__(self): 
        platforms = cl.get_platforms()
        platform = next(p for p in platforms if "Intel" in p.name)
        devices = platform.get_devices()
        self.gpu = next(d for d in devices if cl.device_type.GPU & d.type)
        self.ctx = cl.Context([self.gpu])
        self.queue = cl.CommandQueue(self.ctx)


        self.kernel_code = """
             __kernel void matvec_mul(__global float *matrix, __global float *vector, __global float *result, const int rows, const int cols){
                int i = get_global_id(0); 
                if (i < rows) {
                
                float sum = 0.0f;
                for (int j = 0; j < cols; j++){ 
                    sum += matrix[i * cols + j] * vector[j]; 
                }
                result[i] = sum; 
                }
         } 

            __kernel void dot_product(__global float *v1, __global float *v2, __global float *result, const int size){
            int i = get_global_id(0); 
            if (i < size){
                result[i] = v1[i] * v2[i];
            }
            
            }
        """
            
    
        try: 
            self.program = cl.Program(self.ctx, self.kernel_code).build()
        except Exception as e: 
            print("Kernel build failed", e)
            exit()


    def send_to_gpu(self, data):
        if not isinstance(data, np.ndarray) or data.dtype != np.float32: 
            raise ValueError("Data must be a numpy array of float32")
        
        if data.size == 0: 
            raise ValueError("Cannot send empty array to GPU")
        
        buf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=data.nbytes)
        cl.enqueue_copy(self.queue, buf, data, is_blocking=True)
        return buf

    def receive_from_gpu(self, gpu_data, shape): 
        result = np.empty(shape, dtype=np.float32)
        cl.enqueue_copy(self.queue, result, gpu_data, is_blocking=True)
        return result
    
    def execute_matvec(self, matrix_data, vector_data, rows, cols): 

        matvec_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=rows*4)

        local_size = 64 
        global_size = ((rows + local_size - 1) // local_size) * local_size
        self.program.matvec_mul(self.queue, (global_size,), (local_size,), matrix_data, vector_data, matvec_buf, np.int32(rows), np.int32(cols))
        self.queue.finish()

        return matvec_buf
    
    def execute_dot_product(self, v1, v2, size):

        dot_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=size*4)
        
        local_size = 64
        global_size = ((size + local_size - 1) // local_size) * local_size
        self.program.dot_product(self.queue, (global_size,), (local_size,), v1, v2, dot_buf, np.int32(size))
        self.queue.finish()
        dot_res = self.receive_from_gpu(dot_buf, (size,))

        return np.sum(dot_res)