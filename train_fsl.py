import numpy as np
import torch
from model.trainer.fsl_trainer import FSLTrainer
from model.utils import (
    pprint, set_gpu,
    get_command_line_parser,
    postprocess_args,
)
import random
import os

# from ipdb import launch_ipdb_on_exception
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed(seed)  # GPU
    torch.cuda.manual_seed_all(seed)  # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed)  
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False  
    
    
if __name__ == '__main__':  
    parser = get_command_line_parser()
    args = postprocess_args(parser.parse_args())
    # with launch_ipdb_on_exception():
    pprint(vars(args))

    set_gpu(args.gpu)
    set_seed(args.seed)
    print("Before FSLTrainer")
    trainer = FSLTrainer(args)
    print("Before train")
    trainer.train()
    print("Before evaluate_test")
    trainer.evaluate_test()
    print("Before final_record")
    trainer.final_record()
    print(args.save_path)



