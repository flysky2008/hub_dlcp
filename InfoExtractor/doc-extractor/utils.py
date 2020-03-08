# Version python3.6
# -*- coding: utf-8 -*-
# @Time    : 2019/12/17 10:41
# @Author  : HENRY
# @Email   : mogaoding@163.com
# @File    : doc-extractor
# @Project : dlcp_hub
# @Software: PyCharm

import logging, sys, argparse


def str2bool(v):
    # copy from StackOverflow
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_entity(tag_seq, char_seq):
    EVENT = get_EVENT_entity(tag_seq, char_seq)
    RESULT = get_RESULT_entity(tag_seq, char_seq)
    SOLUTION = get_SOLUTION_entity(tag_seq, char_seq)
    CAUSE = get_CAUSE_entity(tag_seq, char_seq)
    return EVENT, CAUSE, RESULT, SOLUTION


def get_EVENT_entity(tag_seq, char_seq):
    length = len(char_seq)
    EVENT = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-EVENT':
            if 'event' in locals().keys():
                EVENT.append(event)
                del event
            event = char
            if i + 1 == length:
                EVENT.append(event)
        if tag == 'I-EVENT':
            event += char
            if i + 1 == length:
                EVENT.append(event)
        if tag not in ['I-EVENT', 'B-EVENT']:
            if 'event' in locals().keys():
                EVENT.append(event)
                del event
            continue
    return EVENT


def get_CAUSE_entity(tag_seq, char_seq):
    length = len(char_seq)
    CAUSE = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-CAUSE':
            if 'cause' in locals().keys():
                CAUSE.append(cause)
                del cause
            cause = char
            if i + 1 == length:
                CAUSE.append(cause)
        if tag == 'I-CAUSE':
            cause += char
            if i + 1 == length:
                CAUSE.append(cause)
        if tag not in ['I-CAUSE', 'B-CAUSE']:
            if 'cause' in locals().keys():
                CAUSE.append(cause)
                del cause
            continue
    return CAUSE


def get_RESULT_entity(tag_seq, char_seq):
    length = len(char_seq)
    RESULT = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-RESULT':
            if 'result' in locals().keys():
                RESULT.append(result)
                del result
            result = char
            if i + 1 == length:
                RESULT.append(result)
        if tag == 'I-RESULT':
            result += char
            if i + 1 == length:
                RESULT.append(result)
        if tag not in ['I-RESULT', 'B-RESULT']:
            if 'result' in locals().keys():
                RESULT.append(result)
                del result
            continue
    return RESULT


def get_SOLUTION_entity(tag_seq, char_seq):
    length = len(char_seq)
    SOLUTION = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-SOLUTION':
            if 'solution' in locals().keys():
                SOLUTION.append(solution)
                del solution
            solution = char
            if i + 1 == length:
                SOLUTION.append(solution)
        if tag == 'I-SOLUTION':
            solution += char
            if i + 1 == length:
                SOLUTION.append(solution)
        if tag not in ['I-SOLUTION', 'B-SOLUTION']:
            if 'solution' in locals().keys():
                SOLUTION.append(solution)
                del solution
            continue
    return SOLUTION


def get_logger(filename):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    return logger
