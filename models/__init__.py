from .mmotr import build as mmotr_build
from .deformable_mmotr import build as deformable_mmotr_build


def build_model(args):
    if args.model_name == 'mmotr':
        return mmotr_build(args)
    elif args.model_name == 'deformable_mmotr':
        return deformable_mmotr_build(args)
    else:
        raise ValueError(f'Unknown model {args.model_name}')