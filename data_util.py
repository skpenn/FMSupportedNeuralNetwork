import numpy as np
from pandas import read_csv


class Data(object):
    def __init__(self, path: str, batch_size: int=64):
        self._path = path
        self.batch_size = batch_size
        self._feature_size = 0
        self._field_size = 0
        self._feature_labels = None
        self._feature_map = {}
        self._feature_list = []
        self._data_size = 0
        self._x = None
        self._y = None
        self._index = 0

    def load_data(self)->bool:  # Load data from csv file
        try:
            data = read_csv(self._path)
            self._data_size = data.shape[0]
            self._field_size = data.shape[1] - 1
            self._feature_labels = data.columns[1:]

            for label in self._feature_labels:
                distinct_value = data.groupby(label).groups.keys()
                self._feature_list += tuple(distinct_value)
                for value in distinct_value:
                    self._feature_map[label+':'+value] = len(self._feature_map)
            self._feature_size = len(self._feature_map)
            x = []
            for index, row in data.iterrows():
                new_row = []
                for label in self._feature_labels:
                    new_row.append(self._feature_map.get(label+':'+row[label]))
                x.append(new_row)
            #self._x = data.loc[:, self._feature_map].values
            self._x = np.array(x)
            self._y = data.loc[:,data.columns[0]].reshape([self._data_size, 1])

            return True
        except OSError:
            return False

    def get_data_size(self)->int:
        return self._data_size

    def get_field_size(self)->int:
        return self._field_size

    def get_feature_size(self)->int:
        return self._feature_size

    def get_feature_map(self)->tuple:
        return tuple(self._feature_map)

    def get_next_batch(self)->(np.ndarray, np.ndarray):  # generate data batch for model
        if self._index + self.batch_size <= self._data_size:
            x = self._x[self._index:self._index+self.batch_size, ]
            y = self._y[self._index:self._index+self.batch_size, ]
            self._index += self.batch_size
            return x, y
        else:
            align = self._index + self.batch_size - self._data_size
            x1 = self._x[self._index:, ]
            y1 = self._y[self._index:, ]
            x2 = self._x[:align, ]
            y2 = self._y[:align, ]
            self._index = align
            return np.concatenate((x1, x2)), np.concatenate((y1, y2))