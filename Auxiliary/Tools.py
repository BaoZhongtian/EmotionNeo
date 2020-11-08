import os


def SearchFold(loadPath):
    totalPath = []
    for fileName in os.listdir(loadPath):
        if os.path.isfile(os.path.join(loadPath, fileName)):
            totalPath.append(os.path.join(loadPath, fileName))
        else:
            totalPath.extend(SearchFold(os.path.join(loadPath, fileName)))
    return totalPath
