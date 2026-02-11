
import mrcfile, numpy as np
import os

os.makedirs('tomograms', exist_ok=True)
os.makedirs('segmentations', exist_ok=True)

for i in range(100):
    print(i)
    with mrcfile.new(f'tomograms/{i}.mrc', overwrite=True) as f:
        f.set_data(np.random.uniform(0.0, 1.0, (50, 100, 100)).astype(np.float32))
        f.voxel_size = 10.0
    with mrcfile.new(f'segmentations/{i}__microtubule.mrc', overwrite=True) as f:
        f.set_data(np.random.uniform(0.0, 1.0, (50, 100, 100)).astype(np.float32))
        f.voxel_size = 10.0
    with mrcfile.new(f'segmentations/{i}__vault.mrc', overwrite=True) as f:
        f.set_data(np.random.uniform(0.0, 1.0, (50, 100, 100)).astype(np.float32))
        f.voxel_size = 10.0
    with mrcfile.new(f'segmentations/{i}__vimentin.mrc', overwrite=True) as f:
        f.set_data(np.random.uniform(0.0, 1.0, (50, 100, 100)).astype(np.float32))
        f.voxel_size = 10.0
    with mrcfile.new(f'segmentations/{i}__actin.mrc', overwrite=True) as f:
        f.set_data(np.random.uniform(0.0, 1.0, (50, 100, 100)).astype(np.float32))
        f.voxel_size = 10.0
    with mrcfile.new(f'segmentations/{i}__membrane.mrc', overwrite=True) as f:
        f.set_data(np.random.uniform(0.0, 1.0, (50, 100, 100)).astype(np.float32))
        f.voxel_size = 10.0
