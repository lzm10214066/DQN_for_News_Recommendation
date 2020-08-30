import datetime
import os
import shutil
import time

import numpy as np


def printArray(array):
    line = ""
    for v in array:
        line = line + "\t" + str(v)
    print(line)


def shuffle_in_unison_scary(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)


def getDirFiles(path):
    result = []
    if not os.path.exists(path):
        return result
    for file in os.listdir(path):
        path2 = os.path.join(path, file)
        result.append(path2)
    result = sorted(result)
    return result


def delete_file(path):
    if os.path.isfile(path):
        os.remove(path)
    else:
        # for i in os.listdir(path):
        #     path2 = os.path.join(path, i)
        #     delete_file(path2)
        shutil.rmtree(path)


def delete_datas(files, max_file_num):
    if len(files) < max_file_num:
        return
    for i in range(int(max_file_num / 2)):
        delete_file(files[i])


def strToTimeStrap(date):
    date_time = datetime.datetime.strptime(date, "%Y-%m-%d_%H-%M")

    return time.mktime(date_time.timetuple())


def getTrainFiles(files, min_time):
    min_date = strToTimeStrap(min_time)
    result = []
    for file in files:
        #date = file.split("/")[-1]
        _,date=os.path.split(file)
        date_strap = strToTimeStrap(date)
        if date_strap > min_date:
            result.append(file)
    return result


if __name__ == '__main__':
    a = np.array([1, 2, 3])
    b = np.array([-1, -2, -3])
    printArray(a)
    printArray(b)
    shuffle_in_unison_scary(a, b)
    printArray(a)
    printArray(b)
