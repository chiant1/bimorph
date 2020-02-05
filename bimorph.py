import os
import json
import numpy as np
import pandas as pd
import tempfile

from subprocess import Popen, PIPE
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances

def bimorph_json(X:pd.DataFrame, y:pd.Series=None):
    ds = pd.DataFrame()
    ds['dano'] = X.astype(str).apply(lambda x: x.to_dict(), axis=1)
    if y is not None:
        ds['itog'] = y.apply(lambda x: {'y': x})
    return json.dumps(pd.DataFrame(ds).to_dict(orient='records'), ensure_ascii=False)

def bimorph_exec(args, input=None):
    stdin_arg = PIPE if input else None
    process = Popen(args, stdin=stdin_arg, stdout=PIPE)
    output, err = process.communicate(input=input)
    exit_code = process.wait()        
    if exit_code != 0:
        raise ValueError(exit_code)
    return output, err
    
class BimorphClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, sqlite:str=None, e:int=100, r:int=None):
        self.sqlite = sqlite
        self.e = e
        self.r = r
    
    def fit(self, X:pd.DataFrame, y:pd.Series):
        temp = tempfile.NamedTemporaryFile(suffix='.sqlite', prefix='', dir='./').name
        if os.path.isfile(temp):
            os.remove(temp)
        args = ["./bimorph", "-", temp, '-e', str(self.e)]
        output, _ = bimorph_exec(args, input=bimorph_json(X,y).encode("utf-8"))
        self.sqlite = temp
        return BimorphClassifier(sqlite=temp, e=self.e, r=self.r)
    
    def predict(self, X:pd.DataFrame):
        args = ["./bimorph", self.sqlite, "-j", bimorph_json(X)]
        output, err = bimorph_exec(args)
        if len(output) == 0:
            raise ValueError(err)
        result = pd.DataFrame(json.loads(output)).T
        result.index = result.index.astype(np.int)
        result = result.sort_index()
        result.index = X.index
        return result['y']
 