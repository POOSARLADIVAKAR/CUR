#%%
import csv
import numpy as np
import pickle

# %%
def init_matrix():

    '''
    Initializes a matrix with userCount*movieCount shape with zeros.

    Returns:
        ratingsMatrix: zero initialized matrix of required shape
    '''

    movieSet = {}
    userSet = {}

    with open('ratings.csv') as file:
        reader = csv.reader(file)
        for row in reader:
            user = int(row[0])
            movie = int(row[1])
            userSet[user] = 1
            movieSet[movie] = 1

    # print(min(movieSet), max(movieSet))
    # print(min(userSet), max(userSet))
    userCount = max(userSet)
    movieCount = max(movieSet)

    ratingsMatrix = np.zeros((userCount, movieCount))

    return ratingsMatrix

#%%
def populate_matrix(ratingsMatrix):

    '''
    Populates the ratings matrix using the scheme that the rating given by user with userID to movie with movieID is stored at index (userID - 1, movieID - 1) in the ratings matrix.

    Arguments:
        ratingsMatrix: zero initialized ratings matrix
    
    Returns:
        ratingsMatrix: populated ratings matrix
    '''
    
    with open('ratings.csv') as file:
        reader = csv.reader(file)
        for row in reader:
            user = int(row[0]) - 1
            movie = int(row[1]) - 1
            rating = int(row[2])
            ratingsMatrix[user][movie] = rating

    return ratingsMatrix

#%%
def store_matrix(ratingsMatrix):

    '''
    Writes the ratings matrix to persistent storage as a pickle file.

    Arguments:
        ratingsMatrix: the ratings matrix to be stored.
    '''

    with open('ratingsMatrix_noZeros.pickle', 'wb') as file:
        pickle.dump(ratingsMatrix, file)
        file.close()
    # np.save('ratingsMatrix.npy', ratingsMatrix)

#%%
def remove_zeros(ratingsMatrix):

    toBeDeleted = []
    for i in range(ratingsMatrix.shape[1]):
        # print(np.max(ratingsMatrix[:][i]))
        if np.max(ratingsMatrix[:,i]) == 0:
            toBeDeleted.append(i)

    ratingsMatrix = np.delete(ratingsMatrix, toBeDeleted, 1)

    return ratingsMatrix

# %%
# UserIDs   : 1 - 6040
# MovieIDs  : 1 - 3952
# Ratings   : 0 - 5
# userCount : 6040
# movieCount: 3706
# number of entries : 1000209
# size of ratingsMatrix: 190960752B

ratingsMatrix = init_matrix()
(userCount, movieCount) = ratingsMatrix.shape
print(userCount, movieCount)
ratingsMatrix = populate_matrix(ratingsMatrix)
ratingsMatrix = remove_zeros(ratingsMatrix)
store_matrix(ratingsMatrix)

# %%
