import time
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

dataset = ""
imported = False

def readData(fileName, columns, rows=None):
    """Reads a csv file and returns its data.

    filename -- the name of the file to be readed.
    columns  -- the name of the columns to read.
    rows     -- how many rows we want to read (default None = read all)"""
    print('----------')
    print('Reading file:', fileName)
    start = time.time()
    ret = pd.read_csv(fileName, usecols=columns, nrows=rows)
    print('Shape:', ret.shape)
    print('Execution time (s):', time.time()-start)
    return ret

def readTrain(total_loaded, dataDirectory='Archivos/'):
    """Import the 7 features of the train file. Returns a dataFrame or Series (pandas).

    dataDirectory -- directory with the files.
    total_loaded  -- Number of examples to load."""
    columns = ['Semana', 'Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID', 'Demanda_uni_equil']
    dataset = readData(dataDirectory + 'train.csv', columns, rows=total_loaded)
    return dataset

def standardize(array, name):
    """Recieves a dataFrame or Series (from pandas) and returns a numpy array with zero mean and unit variance."""
    # Transform to numpy array
    nparray = array.as_matrix().reshape(array.shape[0],1).astype('float32')
    print('------------')
    print(name)
    print('Different values before:', np.unique(nparray).shape[0])

    # Standardize the data
    nparray = StandardScaler().fit_transform(nparray)

    # Print some information
    print('Mean:', nparray.mean())
    print('Max:', nparray.max())
    print('Min:', nparray.min())
    print('Std:', nparray.std())
    print('Different values after:', np.unique(nparray).shape[0])

    return nparray

def toNparray(series):
    """Recieves a Series (pandas) and returns a numpy array with shape (Series.shape[0],1) and type float32."""
    return series.as_matrix().reshape(series.shape[0], 1).astype('float32')

def getDataset(nExamples):
    """Returns the dataset with nExamples standardized examples from train."""
    global imported, dataset
    if not imported:
        dataset = readTrain(nExamples)
        imported = True

    demanda = toNparray(dataset['Demanda_uni_equil'])
    semana = standardize( dataset['Semana'], 'Semana')
    agencia_id = standardize( dataset['Agencia_ID'], 'Agencia_ID')
    canal_id = standardize( dataset['Canal_ID'], 'Canal_ID')
    ruta_sak = standardize( dataset['Ruta_SAK'], 'Ruta_SAK')
    cliente_id = standardize( dataset['Cliente_ID'], 'Cliente_ID')
    producto_id = standardize( dataset['Producto_ID'], 'Producto_ID')

    return semana, agencia_id, canal_id, ruta_sak, cliente_id, producto_id, demanda

def getRandomDatasets(nExamples, train_size, valid_size, test_size):
    """Returns from the dataset datasets with train_size, valid_size and test_size."""

    semana, agencia_id, canal_id, ruta_sak, cliente_id, producto_id, demanda = getDataset(nExamples)

    joinedDataset = np.concatenate( (semana, agencia_id, canal_id, ruta_sak, cliente_id, producto_id), axis=1)

    # Random rows to be extracted
    train_rows = np.random.choice(joinedDataset.shape[0], size=train_size, replace=False)
    valid_rows = np.random.choice(joinedDataset.shape[0], size=valid_size, replace=False)
    test_rows = np.random.choice(joinedDataset.shape[0], size=test_size, replace=False)

    # 6 feautres
    train_dataset = joinedDataset[train_rows, :]
    valid_dataset = joinedDataset[valid_rows, :]
    test_dataset = joinedDataset[test_rows, :]

    # Outputs
    train_output = demanda[train_rows]
    valid_output = demanda[valid_rows]
    test_output = demanda[test_rows]

    # Print some information
    print('Train dataset', train_dataset.shape)
    print('      outputs', train_output.shape)
    print('Valid dataset', valid_dataset.shape)
    print('      outputs', valid_output.shape)
    print('Test  dataset', test_dataset.shape)
    print('      outputs', test_output.shape)

    return train_dataset, train_output, valid_dataset, valid_output, test_dataset, test_output
