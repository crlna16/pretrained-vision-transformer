#!/usr/bin/env python

def merge_dictionaries_recursively(dict1, dict2):
    ''' Update two config dictionaries recursively.
    adapted from https://jonnyjxn.medium.com/how-to-config-your-machine-learning-experiments-without-the-headaches-bb379de1b957
    Args:
      dict1 (dict): first dictionary to be updated
      dict2 (dict): second dictionary which entries should be preferred
    '''
    if dict2 is None: return

    for k, v in dict2.items():
      if k not in dict1:
        dict1[k] = dict()
      if isinstance(v, dict):
        merge_dictionaries_recursively(dict1[k], v)
      else:
        dict1[k] = v

    return dict1
