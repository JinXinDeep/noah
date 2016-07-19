'''
Created on Jul 16, 2016

@author: lxh5147
'''

import logging

logger = logging.getLogger(__name__)

def check_and_throw_if_fail(condition, msg):
    '''
    condition: boolean; if condition is False, log and throw an exception
    msg: the message to log and exception if condition is False
    '''
    if not condition:
        logger.error(msg)
        raise Exception(msg)


