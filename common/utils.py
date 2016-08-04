'''
Created on Jul 16, 2016

@author: lxh5147
'''

# document style: https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt


def check_and_throw_if_fail(condition, msg):
    '''Throws an exception if the condition does not hold.

    Parameters
    ----------
    condition : bool
    msg: str

    Raises
    ------
    Exception
        If condition is False.
    '''
    if not condition:
        raise Exception(msg)


