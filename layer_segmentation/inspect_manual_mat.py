import scipy.io as sio

mat_path = 'DATA/Manual Segmentation/06 OS - Copy.jpeg_octSegmentation.mat'
data = sio.loadmat(mat_path)
layer = data['imageLayer']
print(f"layer shape: {layer.shape}")
print(f"layer dtype: {layer.dtype}")
print(f"layer[0,0] type: {type(layer[0,0])}")
if hasattr(layer[0,0], 'dtype'):
    print(f"layer[0,0] dtype: {layer[0,0].dtype}")

print(layer[0,0])
