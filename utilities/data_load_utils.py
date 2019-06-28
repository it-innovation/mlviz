import os.path
import pandas as pd

#Class for dictionary with attributes
#See: [https://amir.rachum.com/blog/2016/10/05/python-dynamic-attributes/]
#See: [https://blog.rmotr.com/python-magic-methods-and-getattr-75cf896b3f88]
class AttrDict(dict):
    def __getattr__(self, item):
        return self[item]

def load_dataset(dataset_url, target_names):
    '''
    # Load a csv data file in a SKLearn like data dictionary, with fileds:
    'data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename'.
    (see sklearn.datasetsload_iris())

    #Parameters:
    # dataset_url : a string
    # target_names : list of strings
    '''

    if not os.path.isfile(dataset_url):
            raise FileNotFoundError('File {} not found.'.format(dataset_url))

    df = pd.read_csv(dataset_url)

    if len(target_names) == 0:
        target_names = []
    else:
        target_names = target_names

    for i in target_names:
        if i not in df.columns:
            raise ValueError('Target name {} not in file.'.format(i))

    feature_names = df.columns.values.tolist()
    #remove the target names. list() 'runs' the map at once.
    list(map(feature_names.remove, target_names))

    data = df[feature_names].values
    target = df[target_names].values

    dict_data = AttrDict()
    dict_data["data"] = data
    dict_data["target"] = target
    dict_data["target_names"] = target_names
    dict_data["DESCR"] = ''
    dict_data["feature_names"] = feature_names
    dict_data["filename"] = dataset_url

    #Use a dictionary with attributes as
    # the below dictionary cannot be accessed in an obj.attribute style
    # dict_data = {
    #      "data" : data, 
    #      'target' : target, 
    #      'target_names' : feature_names, 
    #      'DESCR' : '', 
    #      'feature_names' : target_names, 
    #      'filename' : dataset_url
    # }

    return dict_data
