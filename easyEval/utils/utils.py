import re
import time
import os
import numpy as np
import random
import torch
import json
from datetime import datetime

def set_seed(seed=7):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True



def init_output(args):
    try:
        # 创建目录
        os.makedirs("output")
    except FileExistsError:
        pass
    
    current_time = datetime.now().strftime("%Y%m%d%H%M")
    model_name = re.search(r'[^/]+$', args.model).group(0) 
    directory_name = f"output/{model_name}_{args.dataset}_{args.method}_{current_time}/"
    try:
        # 创建目录
        os.makedirs(directory_name)
        print(f"目录 '{directory_name}' 创建成功")
    except FileExistsError:
        print(f"目录 '{directory_name}' 已经存在")
    args.output_dir = directory_name



    







