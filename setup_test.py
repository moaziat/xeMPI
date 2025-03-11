import pyopencl as cl 
import numpy as np


platforms = cl.get_platforms()


if not platforms: 
    print("No opencl platforms found")
    exit()

platform = next(p for p in platforms if "Intel" in p.name)

devices = platform.get_devices()
gpu = next(d for d in devices if cl.device_type.GPU & d.type)

print("Using device:", gpu.name)