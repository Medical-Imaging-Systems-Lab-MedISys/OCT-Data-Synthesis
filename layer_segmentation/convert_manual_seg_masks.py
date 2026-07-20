import os
import glob
import numpy as np
import scipy.io as sio
import cv2

# Colors for layers (BGR)
NR206_COLORS = [
    (0, 0, 0),       # 0: Vitreous (Background)
    (0, 0, 255),     # 1: ILM (Red)
    (0, 128, 128),   # 2: NFL (Olive)
    (0, 255, 255),   # 3: IPL/INL (Yellow)
    (0, 128, 0),     # 4: OPL (DarkGreen)
    (0, 255, 0),     # 5: ONL (BrightGreen)
    (255, 255, 0),   # 6: ELM/IS (Cyan)
    (255, 0, 0),     # 7: OS/RPE (Blue)
    (255, 0, 255)    # 8: RPE/Choroid (Magenta)
]

def create_mask(image_shape, mat_path):
    mask = np.zeros((*image_shape, 3), dtype=np.uint8)
    try:
        data = sio.loadmat(mat_path)
        layer_struct = data['imageLayer'][0,0]
        
        # Extract boundaries
        # Note: the struct fields are tuples or arrays, we need to carefully extract the 1D array
        def get_bound(name):
            try:
                b = layer_struct[name]
                if isinstance(b, np.ndarray) and b.size > 0:
                    return b.flatten()
            except:
                pass
            return None
        
        # The expected order from top to bottom
        bounds = {
            'ilm': get_bound('ilm_0'),
            'nflgcl': get_bound('nflgcl_0'),
            'iplinl': get_bound('iplinl_0'),
            'inlopl': get_bound('inlopl_0'),
            'oplonl': get_bound('oplonl_0'),
            'isos': get_bound('isos_0'),
            'rpe': get_bound('rpe_0')
        }
        
        x_range = get_bound('xrange')
        if x_range is None:
            return mask
            
        h, w = image_shape
        
        for i, x in enumerate(x_range):
            if int(x) < 0 or int(x) >= w:
                continue
            
            x_int = int(x)
            
            # Get y coords for this column
            y_ilm = bounds['ilm'][i] if bounds['ilm'] is not None else h
            y_nflgcl = bounds['nflgcl'][i] if bounds['nflgcl'] is not None else y_ilm
            y_iplinl = bounds['iplinl'][i] if bounds['iplinl'] is not None else y_nflgcl
            y_inlopl = bounds['inlopl'][i] if bounds['inlopl'] is not None else y_iplinl
            y_oplonl = bounds['oplonl'][i] if bounds['oplonl'] is not None else y_inlopl
            y_isos = bounds['isos'][i] if bounds['isos'] is not None else y_oplonl
            y_rpe = bounds['rpe'][i] if bounds['rpe'] is not None else y_isos
            
            y_ilm = np.clip(int(y_ilm), 0, h)
            y_nflgcl = np.clip(int(y_nflgcl), 0, h)
            y_iplinl = np.clip(int(y_iplinl), 0, h)
            y_inlopl = np.clip(int(y_inlopl), 0, h)
            y_oplonl = np.clip(int(y_oplonl), 0, h)
            y_isos = np.clip(int(y_isos), 0, h)
            y_rpe = np.clip(int(y_rpe), 0, h)
            
            # Fill colors
            mask[y_ilm:y_nflgcl, x_int] = NR206_COLORS[1]
            mask[y_nflgcl:y_iplinl, x_int] = NR206_COLORS[2]
            mask[y_iplinl:y_inlopl, x_int] = NR206_COLORS[3]
            mask[y_inlopl:y_oplonl, x_int] = NR206_COLORS[4]
            mask[y_oplonl:y_isos, x_int] = NR206_COLORS[5]
            mask[y_isos:y_rpe, x_int] = NR206_COLORS[6]
            mask[y_rpe:, x_int] = NR206_COLORS[8]
            
    except Exception as e:
        print(f"Error processing {mat_path}: {e}")
        
    return mask

def main():
    ds_dir = "DATA/Manual Segmentation"
    images = glob.glob(os.path.join(ds_dir, "*.jpeg"))
    
    os.makedirs(os.path.join(ds_dir, "masks_gt"), exist_ok=True)
    os.makedirs(os.path.join(ds_dir, "images"), exist_ok=True)
    
    for img_path in images:
        mat_path = img_path + "_octSegmentation.mat"
        if os.path.exists(mat_path):
            img = cv2.imread(img_path)
            if img is not None:
                mask = create_mask(img.shape[:2], mat_path)
                base = os.path.basename(img_path)
                cv2.imwrite(os.path.join(ds_dir, "masks_gt", base.replace('.jpeg', '.png')), mask)
                cv2.imwrite(os.path.join(ds_dir, "images", base), img)
                print(f"Processed {base}")

if __name__ == '__main__':
    main()
