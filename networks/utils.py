import os
import numpy as np

def value_or_default(dictionary, key, default=None):
    ''' check if the key exists in a given dict.
        If exists, return the value; otherwise, return
        the default value if provided.
        If neither the default nor the value are valid,
        raise Exception
    '''
    if key in dictionary:
        return dictionary[key]
    else:
        if default != None:
            return default
        raise ValueError('Key {} not available in dict, and no default value is provided'.format(key))


def get_size(tensor):
    '''
        Extract [Height,Width] of a tensor
    '''
    _,_,h,w = list(tensor.size())
    return [h,w]