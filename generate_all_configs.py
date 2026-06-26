import json
import os
import shutil

p2p_dir = "models/pix2pix"
cgan_dir = "models/cgan_linear"
cnet_dir = "ControlNet"
cfm_dir = "conditional-flow-matching"

for exp_num in [5, 11]:
    src = os.path.join(p2p_dir, f"config_exp{exp_num}.json")
    if not os.path.exists(src):
        continue
    with open(src, 'r') as f:
        config = json.load(f)
        
    # Ensure dataset_path exists for ControlNet
    config["dataset_path"] = "./NR206"
    
    # cGAN
    cgan_cfg = config.copy()
    cgan_cfg["experiment_name"] = cgan_cfg["experiment_name"].replace("Pix2Pix", "cGAN")
    with open(os.path.join(cgan_dir, f"config_exp{exp_num}.json"), 'w') as f:
        json.dump(cgan_cfg, f, indent=2)
        
    # ControlNet
    cnet_cfg = config.copy()
    cnet_cfg["experiment_name"] = cnet_cfg["experiment_name"].replace("Pix2Pix", "ControlNet")
    with open(os.path.join(cnet_dir, f"config_exp{exp_num}.json"), 'w') as f:
        json.dump(cnet_cfg, f, indent=2)
        
    # CFM
    cfm_cfg = config.copy()
    cfm_cfg["experiment_name"] = cfm_cfg["experiment_name"].replace("Pix2Pix", "CFM")
    with open(os.path.join(cfm_dir, f"config_exp{exp_num}.json"), 'w') as f:
        json.dump(cfm_cfg, f, indent=2)

print("Configs generated successfully.")
