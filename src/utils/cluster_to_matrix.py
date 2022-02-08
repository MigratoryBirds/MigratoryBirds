import numpy as np

def find_mini(axis):
    mini = np.max(locations[:,axis]) + 1
    for i in locations[:,axis]:
        for ii in locations[:,axis]:
            if (np.abs(i - ii) < mini) and (np.abs(i - ii) != 0):
                mini = np.abs(i - ii)
    return mini


def create_matrix(locations, mini_dist = 0, returnMini = False):

    # min distance between points:
    mini = np.min([find_mini(0), find_mini(1)]) if mini_dist == 0 else mini_dist
    
    if returnMini:
        return mini

    # size of the matrix:
    x_dist = np.max(locations[:,0]) - np.min(locations[:,0]) + mini
    y_dist = np.max(locations[:,1]) - np.min(locations[:,1]) + mini

    # create matrix of zeros:
    matrix = np.zeros([int(x_dist/mini), int(y_dist/mini)])

    # split the x- and y-axis:
    loc_i = np.min(locations[:,0]) + mini/2 + np.arange(int(x_dist/mini))*mini
    loc_j = np.min(locations[:,1]) + mini/2 + np.arange(int(y_dist/mini))*mini

    # fill the matrix:
    added = np.array([])

    for i in range(len(loc_i)):
        for j in range(len(loc_j)):
            for ll in range(locations.shape[0]):
                if (locations[ll,0] < loc_i[i]) and (locations[ll,1] < loc_j[j]) and (ll not in added):
                    added = np.append(added, ll)
                    matrix[i, j] = 1
                    
    return matrix