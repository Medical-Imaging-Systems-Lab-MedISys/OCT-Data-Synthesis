import scipy.io as sio
paths = [
    '/data/vds/mmk/Codes/oct_data_synthesis/DATA/2011_IOVS_Chiu/Chiu_IOVS_2011/Automatic versus Manual Study/Group3_Volume5.mat'
]
for p in paths:
    try:
        data = sio.loadmat(p)
        print(f'\n--- {p} ---')
        for k, v in data.items():
            if not k.startswith('__'):
                print(f'{k}: type={type(v)}, shape={getattr(v, "shape", "N/A")}')
    except Exception as e:
        print(f'\n--- {p} --- Error: {e}')
