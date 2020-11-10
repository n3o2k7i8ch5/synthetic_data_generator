import os


def parent_path():
    path = os.getcwd()
    elements = path.split('/')

    p_path = ''
    for i in range(len(elements) - 1):
        p_path += elements[i] + '/'

    return p_path


HIST_POINTS = 100
BATCH_SIZE = 128*8
INP_RAND_SIZE = 10
