from datasets.tunnel import TunnelDataset
from datasets.ua_detrac import UADETRAC


def get_parser_data_from_dataset(dataset):
    """The closure function return a func that return parser class and item from dataset by dataset_item or sequence_name."""
    def get_parser(dataset_item=None, sequence_name=None):
        if sequence_name is not None:
            return dataset.get_parser_from_name(sequence_name)
        elif dataset_item is not None:
            return dataset.get_parser_from_item(dataset_item)
    
    return get_parser


def build_dataset(image_set, dataset_name, **kwargs):
    if dataset_name == 'tunnel':
        return TunnelDataset(image_set, **kwargs)
    elif dataset_name == 'ua':
        return UADETRAC(image_set, **kwargs)
    raise ValueError(f'dataset {dataset_name} not supported')
