# Retinal OCT Synthetic Data Generation - MLflow CFM Runs

This table summarizes the results of the **Conditional Flow Matching (CFM)** training runs logged in MLflow.

| Experiment Name / Run Name | Loss Type | Train Loss | Val Loss | Key Configuration | Description / Notes | MLflow Link |
|---|---|---|---|---|---|---|
| **cfm_2026-06-26_14-13-34_exp1**<br>cfm_2026-06-26_14-13-39 | MSE (Vector Field) | 0.1848 | N/A | Batch: 16<br>LR: 0.0002<br>Epochs: 100<br>Size: 256x256<br>Steps: 50<br>In Channels: 2 | **Run:** cfm_2026-06-26_14-13-39<br>**Configuration Summary:**<br>- **Resolution:** 256x256<br>- **Batch Size:** 16<br>- **Epochs:** 100<br>- **Learning Rate:** 0.0002 | [View Run](https://dagshub.com/IISc-MedISys/OCT-Data-Synthesis.mlflow/#/experiments/39/runs/b451558d3ecf457693685b5defb470bf) |
| **cfm_2026-06-26_11-46-18_exp1**<br>sincere-lamb-969 | MSE (Vector Field) | 0.0011 | 0.0014 | Batch: 16<br>LR: 0.0002<br>Epochs: 100<br>Size: 256x256<br>Steps: 50<br>In Channels: 2 | **Run:** sincere-lamb-969<br>**Configuration Summary:**<br>- **Resolution:** 256x256<br>- **Batch Size:** 16<br>- **Epochs:** 100<br>- **Learning Rate:** 0.0002 | [View Run](https://dagshub.com/IISc-MedISys/OCT-Data-Synthesis.mlflow/#/experiments/38/runs/ceec511a01434681825d5eec4aabc824) |
| **CFM_V2_OnlineAug_Exp5_FastLR**<br>CFM_V2_OnlineAug_Exp5_FastLR_2026-06-25_17-47-43 | L1 Penalty | 0.0881 | 0.0721 | Batch: 16<br>LR: 0.001<br>Epochs: 100<br>Size: 256x256 | **Run:** CFM_V2_OnlineAug_Exp5_FastLR_2026-06-25_17-47-43<br>**Configuration Summary:**<br>- **Resolution:** 256x256<br>- **Batch Size:** 16<br>- **Epochs:** 100<br>- **Learning Rate:** 0.001 | [View Run](https://dagshub.com/IISc-MedISys/OCT-Data-Synthesis.mlflow/#/experiments/15/runs/eb1d883696894e8bbb5cac8c609e5d82) |
