# %%
import pandas as pd
import re

df_result = []
dataset_list = ['curvefault_a_val',
                'curvefault_b_val',
                'curvevel_a_val',
                'curvevel_b_val',
                'flatfault_a_val',
                'flatfault_b_val',
                'flatvel_a_val',
                'flatvel_b_val',
                'style_a_val',
                'style_b_val',]
result_path_list=['/workspace/OpenFWI-main/moz_work/InversionNet/bigfwi_train/default',
             '/workspace/OpenFWI-main/moz_work/InversionNet/bigfwi_train/flip',]




dataset=dataset_list[0]

for result_path in result_path_list:
    for dataset in dataset_list:
        with open(result_path + '/' + dataset + '/metrics.txt', 'r') as f:
            text = f.read()
        model = result_path.split('/')[-3]
        suffix = result_path.split('/')[-1]
        # 正規表現で数値を抽出
        result = {
            'model': model,
            'dataset': dataset,
            'suffix': suffix,
            'MAE': float(re.search(r'MAE:\s*([0-9.e+-]+)', text).group(1)),
            'MSE': float(re.search(r'MSE:\s*([0-9.e+-]+)', text).group(1)),
            'SSIM': float(re.search(r'SSIM:\s*([0-9.e+-]+)',3text).group(1)),
            'Velocity MAE': float(re.search(r'Velocity MAE:\s*([0-9.e+-]+)', text).group(1)),
            'Velocity MSE': float(re.search(r'Velocity MSE:\s*([0-9.e+-]+)', text).group(1)),
        }
        df_result.append(result)
df_result = pd.DataFrame(df_result)

# %%
