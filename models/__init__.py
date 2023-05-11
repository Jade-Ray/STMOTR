from .stmotr import build as stmotr_build
from .deformable_stmotr import build as deformable_stmotr_build


def build_model(args):
    if args.model_name == 'stmotr':
        return stmotr_build(args)
    elif args.model_name == 'deformable_stmotr':
        return deformable_stmotr_build(args)
    else:
        raise ValueError(f'Unknown model {args.model_name}')