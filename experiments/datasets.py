# 
# Nathan Lay
# AI Resource at National Cancer Institute
# National Institutes of Health
# January 2021
# 
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR(S) ``AS IS'' AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE AUTHOR(S) BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 

import os
import requests
import sklearn.datasets
import numpy as np
import csv
import hashlib
import zipfile
import tarfile
import gzip

datasets = dict()

def _make_data_cache_dir(name):
    if not os.path.exists(".datacache"):
        os.mkdir(".datacache")
        
    path = os.path.join(".datacache", name)
    
    if not os.path.exists(path):
        os.mkdir(path)
        
    return path
    
def _hash_file(path):
    hasher = hashlib.sha256()
    
    with open(path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)

    return hasher.hexdigest()
    
def _download_file(srcUrl, destFile):
    with requests.get(srcUrl) as req:
        with open(destFile, "wb") as f:
            f.write(req.content)

_abalone_xtrain = None
_abalone_ytrain = None
_abalone_xtest = None
_abalone_ytest = None

def load_abalone_reg():
    global _abalone_xtrain, _abalone_ytrain, _abalone_xtest, _abalone_ytest

    if _abalone_xtrain is not None:
        return _abalone_xtrain.copy(), _abalone_ytrain.copy(), _abalone_xtest.copy(), _abalone_ytest.copy()

    path = _make_data_cache_dir("abalone")
    
    baseUrl = "https://archive.ics.uci.edu/ml/machine-learning-databases"
    files = [ "abalone.data", "abalone.names" ]
    hashes = [ "de37cdcdcaaa50c309d514f248f7c2302a5f1f88c168905eba23fe2fbc78449f", None ]
    
    for file, expectedHash in zip(files, hashes):
        srcUrl = baseUrl + "/abalone/" + file
        destFile = os.path.join(path, file)
        
        if not os.path.exists(destFile):
            _download_file(srcUrl, destFile)
            
        tmpHash = _hash_file(destFile)
        
        if expectedHash is not None and expectedHash != tmpHash:
            raise Exception(f"Error: Hash mismatch for '{destFile}'. Expected '{expectedHash}' but got '{tmpHash}'")
    
    data = np.zeros([4177, 8])
    target = np.zeros([4177])
    
    mapper = [lambda x : float(x)]*9
    mapper[0] = lambda x : float([ 'M', 'F', 'I' ].index(x))
    
    with open(os.path.join(path, "abalone.data"), newline="") as f:
        reader = csv.reader(f, delimiter=',')
        
        for row in reader:
            for token, i in zip(row, range(9)):
                if i == 8:
                    target[reader.line_num-1] = mapper[i](token)
                else:
                    data[reader.line_num-1, i] = mapper[i](token) 

    # Split defined by the paper in the abalone.names file
    _abalone_xtrain = data[:3133,:]
    _abalone_ytrain = target[:3133]
    _abalone_xtest = data[3133:,:]
    _abalone_ytest = target[3133:]

    return _abalone_xtrain.copy(), _abalone_ytrain.copy(), _abalone_xtest.copy(), _abalone_ytest.copy()
    #return data[:3133,:], target[:3133], data[3133:,:], target[3133:]

def load_abalone():
    xtrain, ytrain, xtest, ytest = load_abalone_reg()

    # Check abalone.names file for the paper that defines this label mapping
    mapfn = np.vectorize(lambda x : 0.0 if float(x) < 9 else ( 1.0 if float(x) < 11 else 2.0 ))
    ytrain = mapfn(ytrain)
    ytest = mapfn(ytest)
    
    return xtrain, ytrain, xtest, ytest
    
_cifar10_xtrain = None
_cifar10_ytrain = None
_cifar10_xtest = None
_cifar10_ytest = None

def load_cifar10():
    global _cifar10_xtrain, _cifar10_ytrain, _cifar10_xtest, _cifar10_ytest

    if _cifar10_xtrain is not None:
        return _cifar10_xtrain.copy(), _cifar10_ytrain.copy(), _cifar10_xtest.copy(), _cifar10_ytest.copy()

    path = _make_data_cache_dir("cifar10")

    baseUrl = "https://www.cs.toronto.edu/~kriz"
    files = [ "cifar-10-binary.tar.gz" ]
    hashes = [ "c4a38c50a1bc5f3a1c5537f2155ab9d68f9f25eb1ed8d9ddda3db29a59bca1dd" ]

    for file, expectedHash in zip(files, hashes):
        srcUrl = baseUrl + '/' + file
        destFile = os.path.join(path, file)
        
        if not os.path.exists(destFile):
            _download_file(srcUrl, destFile)
            
        tmpHash = _hash_file(destFile)
        
        if expectedHash is not None and expectedHash != tmpHash:
            raise Exception(f"Error: Hash mismatch for '{destFile}'. Expected '{expectedHash}' but got '{tmpHash}'")
   
    dataPath = os.path.join(path, "cifar-10-batches-bin")
    
    if not os.path.exists(dataPath):
        with gzip.open(os.path.join(path, "cifar-10-binary.tar.gz"), "rb") as fz:
            with tarfile.TarFile(fileobj=fz, mode='r') as ft:
                ft.extractall(path)

    batchFiles = [ "data_batch_1.bin", "data_batch_2.bin", "data_batch_3.bin", "data_batch_4.bin", "data_batch_5.bin", "test_batch.bin" ]

    xtrain = np.zeros([50000, 3072])
    ytrain = np.zeros([50000])
    xtest = np.zeros([10000, 3072])
    ytest = np.zeros([10000])

    i = 0
    for batchFile in batchFiles:
        batchFile = os.path.join(dataPath, batchFile)

        with open(batchFile, "rb") as f:
            raw = np.frombuffer(f.read(10000*3073), dtype=np.uint8)
            raw = np.reshape(raw, [-1, 3073])
            labels = raw[:, 0].astype(np.float32)
            data = raw[:, 1:].astype(np.float32)

            if i < 5:
                begin = i*10000
                end = begin + 10000

                xtrain[begin:end, :] = data
                ytrain[begin:end] = labels
            else:
                begin = 0
                end = 10000

                xtest[begin:end, :] = data
                ytest[begin:end] = labels

        i += 1

    xtrain = np.reshape(xtrain, [-1, 3, 32, 32])
    xtest = np.reshape(xtest, [-1, 3, 32, 32])

    _cifar10_xtrain = xtrain
    _cifar10_ytrain = ytrain
    _cifar10_xtest = xtest
    _cifar10_ytest = ytest

    return _cifar10_xtrain.copy(), _cifar10_ytrain.copy(), _cifar10_xtest.copy(), _cifar10_ytest.copy()

_har_xtrain = None
_har_ytrain = None
_har_xtest = None
_har_ytest = None

def load_har():
    global _har_xtrain, _har_ytrain, _har_xtest, _har_ytest

    if _har_xtrain is not None:
        return _har_xtrain.copy(), _har_ytrain.copy(), _har_xtest.copy(), _har_ytest.copy()

    path = _make_data_cache_dir("har")
    
    baseUrl = "https://archive.ics.uci.edu/ml/machine-learning-databases"
    files = [ "UCI HAR Dataset.zip", "UCI HAR Dataset.names"  ]
    hashes = [ "2045e435c955214b38145fb5fa00776c72814f01b203fec405152dac7d5bfeb0", None ]
    
    for file, expectedHash in zip(files, hashes):
        srcUrl = baseUrl + "/00240/" + file
        destFile = os.path.join(path, file)
        
        if not os.path.exists(destFile):
            _download_file(srcUrl, destFile)
            
        tmpHash = _hash_file(destFile)
        
        if expectedHash is not None and expectedHash != tmpHash:
            raise Exception(f"Error: Hash mismatch for '{destFile}'. Expected '{expectedHash}' but got '{tmpHash}'")
    
    dataPath = os.path.join(path, "UCI HAR Dataset")
    
    if not os.path.exists(dataPath):
        with zipfile.ZipFile(os.path.join(path, "UCI HAR Dataset.zip"), "r") as f:
            f.extractall(path)

    trainFileX = os.path.join(dataPath, "train", "X_train.txt")
    trainFileY = os.path.join(dataPath, "train", "y_train.txt")
    
    testFileX = os.path.join(dataPath, "test", "X_test.txt")
    testFileY = os.path.join(dataPath, "test", "y_test.txt")
    
    xtrain = np.zeros([7352, 561])
    ytrain = np.zeros([7352])
    
    xtest = np.zeros([2947, 561])
    ytest = np.zeros([2947])
    
    with open(trainFileX, newline="") as f:
        for line, line_num in zip(f, range(7532)):
            values = [ float(k) for k in line.split(" ") if len(k) > 0 ]
            for value, i in zip(values, range(561)):
                xtrain[line_num, i] = value
                
    with open(trainFileY, newline="") as f:
        for y, line_num in zip(f, range(7352)):
            ytrain[line_num] = y
                
    with open(testFileX, newline="") as f:
        for line, line_num in zip(f, range(2947)):
            values = [ float(k) for k in line.split(" ") if len(k) > 0 ]
            for value, i in zip(values, range(561)):
                xtest[line_num, i] = value

    with open(testFileY, newline="") as f:
        for y, line_num in zip(f, range(2947)):
            ytest[line_num] = y
    
    _har_xtrain = xtrain
    _har_ytrain = ytrain-1
    _har_xtest = xtest
    _har_ytest = ytest-1

    return _har_xtrain.copy(), _har_ytrain.copy(), _har_xtest.copy(), _har_ytest.copy()
    #return xtrain, ytrain-1, xtest, ytest-1
    
def load_iris():
    # Use sklearn when possible!
    data = sklearn.datasets.load_iris()
    return data.data, data.target

def load_letter():
    path = _make_data_cache_dir("letter")

    baseUrl = "https://archive.ics.uci.edu/ml/machine-learning-databases"
    files = [ "letter-recognition.data", "letter-recognition.names" ]
    hashes = [ "2b89f3602cf768d3c8355267d2f13f2417809e101fc2b5ceee10db19a60de6e2", None ]

    for file, expectedHash in zip(files, hashes):
        srcUrl = baseUrl + "/letter-recognition/" + file
        destFile = os.path.join(path, file)
        
        if not os.path.exists(destFile):
            _download_file(srcUrl, destFile)
            
        tmpHash = _hash_file(destFile)
        
        if expectedHash is not None and expectedHash != tmpHash:
            raise Exception(f"Error: Hash mismatch for '{destFile}'. Expected '{expectedHash}' but got '{tmpHash}'")

    data = np.zeros([20000,16])
    targets = np.zeros([20000])
    
    mapper = [lambda x : float(x)]*17
    mapper[0] = lambda x : ord(x) - ord('A')
    
    with open(os.path.join(path, "letter-recognition.data"), newline="") as f:
        reader = csv.reader(f, delimiter=',')
        
        for row in reader:
            for token, i in zip(row, range(17)):
                if i == 0:
                    targets[reader.line_num-1] = mapper[i](token)
                else:
                    data[reader.line_num-1, i-1] = mapper[i](token)
                    
    return data[:16000, :], targets[:16000], data[16000:, :], targets[16000:]

_madelon_xtrain = None
_madelon_ytrain = None
_madelon_xtest = None
_madelon_ytest = None
    
def load_madelon():
    global _madelon_xtrain, _madelon_ytrain, _madelon_xtest, _madelon_ytest

    if _madelon_xtrain is not None:
        return _madelon_xtrain.copy(), _madelon_ytrain.copy(), _madelon_xtest.copy(), _madelon_ytest.copy()

    path = _make_data_cache_dir("madelon")

    baseUrl = "https://archive.ics.uci.edu/ml/machine-learning-databases"
    files = [ "MADELON/madelon_train.data", "MADELON/madelon_train.labels", "MADELON/madelon_valid.data", "madelon_valid.labels", "Dataset.pdf" ]
    hashes = [ "0b6e37711efdc7ab74250c51e7d6966bffd9f8f2b0e9971e882c3a7a380c2314", "10bffcc3b017467cb810d326b931951adcbde2b4870faf981da2991993b48689", "cb9a46475147de1a58d9f15ef4c318e5b85de344e0f9ab92626fa446aa815dc9", "fbac32fa87fac41ab731994a1c824e7b25396069bff6df242c01fe240fcb34dc", None ]

    for file, expectedHash in zip(files, hashes):
        srcUrl = baseUrl + "/madelon/" + file
        destFile = os.path.join(path, os.path.basename(file))
        
        if not os.path.exists(destFile):
            _download_file(srcUrl, destFile)
            
        tmpHash = _hash_file(destFile)
        
        if expectedHash is not None and expectedHash != tmpHash:
            raise Exception(f"Error: Hash mismatch for '{destFile}'. Expected '{expectedHash}' but got '{tmpHash}'")

    trainFileX = os.path.join(path, "madelon_train.data")
    trainFileY = os.path.join(path, "madelon_train.labels")
    
    testFileX = os.path.join(path, "madelon_valid.data")
    testFileY = os.path.join(path, "madelon_valid.labels")
    
    xtrain = np.zeros([2000, 500])
    ytrain = np.zeros([2000])
    
    xtest = np.zeros([600, 500])
    ytest = np.zeros([600])
    
    with open(trainFileX, newline="") as f:
        for line, line_num in zip(f, range(2000)):
            values = [ float(k) for k in line.strip().split(" ") if len(k) > 0 ]
            xtrain[line_num, :] = values
            #for value, i in zip(values, range(500)):
                #xtrain[line_num, i] = value
                
    with open(trainFileY, newline="") as f:
        for y, line_num in zip(f, range(2000)):
            ytrain[line_num] = 0.0 if float(y) < 0.0 else 1.0
                
    with open(testFileX, newline="") as f:
        for line, line_num in zip(f, range(600)):
            values = [ float(k) for k in line.strip().split(" ") if len(k) > 0 ]
            xtest[line_num, :] = values
            #for value, i in zip(values, range(500)):
                #xtest[line_num, i] = value

    with open(testFileY, newline="") as f:
        for y, line_num in zip(f, range(600)):
            ytest[line_num] = 0.0 if float(y) < 0.0 else 1.0

    _madelon_xtrain = xtrain
    _madelon_ytrain = ytrain
    _madelon_xtest = xtest
    _madelon_ytest = ytest
    
    return _madelon_xtrain.copy(), _madelon_ytrain.copy(), _madelon_xtest.copy(), _madelon_ytest.copy()

_mnist_xtrain = None
_mnist_ytrain = None
_mnist_xtest = None
_mnist_ytest = None
    
def load_mnist():
    global _mnist_xtrain, _mnist_ytrain, _mnist_xtest, _mnist_ytest

    if _mnist_xtrain is not None:
        return _mnist_xtrain.copy(), _mnist_ytrain.copy(), _mnist_xtest.copy(), _mnist_ytest.copy()

    path = _make_data_cache_dir("mnist")

    baseUrl = "http://yann.lecun.com/exdb/mnist"
    files = [ "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz" ]
    hashes = [ "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609", "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c", "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6", "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6" ]
    
    for file, expectedHash in zip(files, hashes):
        srcUrl = baseUrl + '/' + file
        destFile = os.path.join(path, file)
        
        if not os.path.exists(destFile):
            _download_file(srcUrl, destFile)
            
        tmpHash = _hash_file(destFile)
        
        if expectedHash is not None and expectedHash != tmpHash:
            raise Exception(f"Error: Hash mismatch for '{destFile}'. Expected '{expectedHash}' but got '{tmpHash}'")
    
    trainFileX = os.path.join(path, "train-images-idx3-ubyte.gz")
    trainFileY = os.path.join(path, "train-labels-idx1-ubyte.gz")
    
    testFileX = os.path.join(path, "t10k-images-idx3-ubyte.gz")
    testFileY = os.path.join(path, "t10k-labels-idx1-ubyte.gz")
    
    with gzip.open(trainFileX, mode="rb") as f:
        magic = int.from_bytes(f.read(4), byteorder="big")
        length = int.from_bytes(f.read(4), byteorder="big")
        height = int.from_bytes(f.read(4), byteorder="big")
        width = int.from_bytes(f.read(4), byteorder="big")

        if magic != 2051 or length != 60000 or width != 28 or height != 28:
          raise Exception(f"Error: Unexpected file contents in '{trainFileX}'.")

        xtrain = np.frombuffer(f.read(length*width*height), dtype=np.uint8).astype(np.float32)
        xtrain = np.reshape(xtrain, [length, 1, width, height]) # Stored row-wise
                
    with gzip.open(trainFileY, mode="rb") as f:
        magic = int.from_bytes(f.read(4), byteorder="big")
        length = int.from_bytes(f.read(4), byteorder="big")

        if magic != 2049 or length != 60000:
          raise Exception(f"Error: Unexpected file contents in '{trainFileY}'.")

        ytrain = np.frombuffer(f.read(length), dtype=np.uint8).astype(np.float32)
                
    with gzip.open(testFileX, mode="rb") as f:
        magic = int.from_bytes(f.read(4), byteorder="big")
        length = int.from_bytes(f.read(4), byteorder="big")
        height = int.from_bytes(f.read(4), byteorder="big")
        width = int.from_bytes(f.read(4), byteorder="big")

        if magic != 2051 or length != 10000 or width != 28 or height != 28:
          raise Exception(f"Error: Unexpected file contents in '{testFileX}'.")

        xtest = np.frombuffer(f.read(length*width*height), dtype=np.uint8).astype(np.float32)
        xtest = np.reshape(xtest, [length, 1, width, height]) # Stored row-wise

    with gzip.open(testFileY, mode="rb") as f:
        magic = int.from_bytes(f.read(4), byteorder="big")
        length = int.from_bytes(f.read(4), byteorder="big")

        if magic != 2049 or length != 10000:
          raise Exception(f"Error: Unexpected file contents in '{testFileY}'.")

        ytest = np.frombuffer(f.read(length), dtype=np.uint8).astype(np.float32)

    _mnist_xtrain = xtrain
    _mnist_ytrain = ytrain
    _mnist_xtest = xtest
    _mnist_ytest = ytest

    return _mnist_xtrain.copy(), _mnist_ytrain.copy(), _mnist_xtest.copy(), _mnist_ytest.copy()

_poker_xtrain = None
_poker_ytrain = None
_poker_xtest = None
_poker_ytest = None

def load_poker():
    global _poker_xtrain, _poker_ytrain, _poker_xtest, _poker_ytest

    if _poker_xtrain is not None:
        return _poker_xtrain.copy(), _poker_ytrain.copy(), _poker_xtest.copy(), _poker_ytest.copy()

    path = _make_data_cache_dir("poker")

    baseUrl = "https://archive.ics.uci.edu/ml/machine-learning-databases"
    files = [ "poker-hand-training-true.data", "poker-hand-testing.data", "poker-hand.names" ]
    hashes = [ "37becdf87d5f8cbf2b91d6471e965a25b86cb4a6d878c0f94a4025969fca464f", "3cd75958e19dd321ed5ca3f7f154c0f6aad544aab9f37731ac545b5f66b232c7", None ]
    
    for file, expectedHash in zip(files, hashes):
        srcUrl = baseUrl + "/poker/" + file
        destFile = os.path.join(path, file)
        
        if not os.path.exists(destFile):
            _download_file(srcUrl, destFile)
            
        tmpHash = _hash_file(destFile)
        
        if expectedHash is not None and expectedHash != tmpHash:
            raise Exception(f"Error: Hash mismatch for '{destFile}'. Expected '{expectedHash}' but got '{tmpHash}'")
    
    xtrain = np.zeros([25010, 10])
    ytrain = np.zeros([25010])
    
    xtest = np.zeros([1000000, 10])
    ytest = np.zeros([1000000])
    
    mapper = [lambda x : float(x)]*11
    
    with open(os.path.join(path, "poker-hand-training-true.data"), newline="") as f:
        reader = csv.reader(f, delimiter=',')
        
        for row in reader:
            for token, i in zip(row, range(11)):
                if i == 10:
                    ytrain[reader.line_num-1] = mapper[i](token)
                else:
                    xtrain[reader.line_num-1, i] = mapper[i](token)
                    
    with open(os.path.join(path, "poker-hand-testing.data"), newline="") as f:
        reader = csv.reader(f, delimiter=',')
        
        for row in reader:
            for token, i in zip(row, range(11)):
                if i == 10:
                    ytest[reader.line_num-1] = mapper[i](token)
                else:
                    xtest[reader.line_num-1, i] = mapper[i](token)

    _poker_xtrain = xtrain                    
    _poker_ytrain = ytrain
    _poker_xtest = xtest
    _poker_ytest = ytest

    return _poker_xtrain.copy(), _poker_ytrain.copy(), _poker_xtest.copy(), _poker_ytest.copy()

_usps_xtrain = None
_usps_ytrain = None
_usps_xtest = None
_usps_ytest = None

def load_usps():
    global _usps_xtrain, _usps_ytrain, _usps_xtest, _usps_ytest

    if _usps_xtrain is not None:
        return _usps_xtrain.copy(), _usps_ytrain.copy(), _usps_xtest.copy(), _usps_ytest.copy()

    path = os.path.dirname(__file__)
    path = os.path.join(path, "data", "usps")
    
    trainFile = os.path.join(path, "zip.train.gz")
    testFile = os.path.join(path, "zip.test.gz")

    xtrain = np.zeros([7291, 16*16])
    ytrain = np.zeros([7291])

    xtest = np.zeros([2007, 16*16])
    ytest = np.zeros([2007])

    with gzip.open(trainFile, mode="rt", newline="") as f:
        for line, line_num in zip(f, range(7291)):
            values = [ float(x) for x in line.strip().split(" ") ]
            ytrain[line_num] = values[0]
            xtrain[line_num, :] = values[1:]
       

    with gzip.open(testFile, mode="rt", newline="") as f:
        for line, line_num in zip(f, range(2007)):
            values = [ float(x) for x in line.strip().split(" ") ]
            ytest[line_num] = values[0]
            xtest[line_num, :] = values[1:]

    xtrain = np.reshape(xtrain, [7291, 1, 16, 16])
    xtest = np.reshape(xtest, [2007, 1, 16, 16])

    _usps_xtrain = xtrain
    _usps_ytrain = ytrain
    _usps_xtest = xtest
    _usps_ytest = ytest

    return _usps_xtrain.copy(), _usps_ytrain.copy(), _usps_xtest.copy(), _usps_ytest.copy()

_forest_cover_xtrain = None
_forest_cover_ytrain = None
_forest_cover_xtest = None
_forest_cover_ytest = None

def load_forest_cover():
    global _forest_cover_xtrain, _forest_cover_ytrain, _forest_cover_xtest, _forest_cover_ytest

    if _forest_cover_xtrain is not None:
        return _forest_cover_xtrain.copy(), _forest_cover_ytrain.copy(), _forest_cover_xtest.copy(), _forest_cover_ytest.copy()

    path = _make_data_cache_dir("forest_cover")

    baseUrl = "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    files = [ "covtype.data.gz", "covtype.info", "old_covtype.info" ]
    hashes = [ "614360d0257557dd1792834a85a1cdebfadc3c4f30b011d56afee7ffb5b15771", None, None ]
    
    for file, expectedHash in zip(files, hashes):
        srcUrl = baseUrl + "/covtype/" + file
        destFile = os.path.join(path, file)
        
        if not os.path.exists(destFile):
            _download_file(srcUrl, destFile)
            
        tmpHash = _hash_file(destFile)
        
        if expectedHash is not None and expectedHash != tmpHash:
            raise Exception(f"Error: Hash mismatch for '{destFile}'. Expected '{expectedHash}' but got '{tmpHash}'")

    xdata = np.zeros([581012,54])
    ydata = np.zeros([581012])

    with gzip.open(os.path.join(path, "covtype.data.gz"), mode="rt", newline="") as f:
        reader = csv.reader(f, delimiter=',')
        
        for row in reader:
            values = [ float(token) for token in row ]
            xdata[reader.line_num-1, :] = values[:54]
            ydata[reader.line_num-1] = values[54] - 1

    _forest_cover_xtrain = xdata[:15120,:]
    _forest_cover_ytrain = ydata[:15120]
    _forest_cover_xtest = xdata[15120:,:]
    _forest_cover_ytest = ydata[15120:]

    return _forest_cover_xtrain.copy(), _forest_cover_ytrain.copy(), _forest_cover_xtest.copy(), _forest_cover_ytest.copy()
    #return xdata[:15120,:], ydata[:15120], xdata[15120:,:], ydata[15120:]

def load_breast_cancer():
    # Use sklearn when possible!
    data = sklearn.datasets.load_breast_cancer()
    return data.data, data.target

# This probably doesn't need to be cached but let's prevent IO errors on biowulf anyway...
_pima_xdata = None
_pima_ydata = None

def load_pima():
    global _pima_xdata, _pima_ydata

    if _pima_xdata is not None:
        return _pima_xdata.copy(), _pima_ydata.copy()

    path = os.path.dirname(__file__)
    path = os.path.join(path, "data", "pima")

    xdata = np.zeros([768, 8])
    ydata = np.zeros([768])

    with open(os.path.join(path, "pima-indians-diabetes.data"), newline="") as f:
        reader = csv.reader(f, delimiter=',')
        
        for row in reader:
            for token, i in zip(row, range(9)):
                if i == 8:
                    ydata[reader.line_num-1] = float(token)
                else:
                    xdata[reader.line_num-1, i] = float(token)
   
    _pima_xdata = xdata
    _pima_ydata = ydata

    return _pima_xdata.copy(), _pima_ydata.copy()

_car_evaluation_xdata = None
_car_evaluation_ydata = None

def load_car_evaluation():
    global _car_evaluation_xdata, _car_evaluation_ydata

    if _car_evaluation_xdata is not None:
        return _car_evaluation_xdata.copy(), _car_evaluation_ydata.copy()

    path = _make_data_cache_dir("car_evaluation")

    baseUrl = "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    files = [ "car.data", "car.names" ]
    hashes = [ "b703a9ac69f11e64ce8c223c0a40de4d2e9d769f7fb20be5f8f2e8a619893d83", None ]
    
    for file, expectedHash in zip(files, hashes):
        srcUrl = baseUrl + "/car/" + file
        destFile = os.path.join(path, file)
        
        if not os.path.exists(destFile):
            _download_file(srcUrl, destFile)
            
        tmpHash = _hash_file(destFile)
        
        if expectedHash is not None and expectedHash != tmpHash:
            raise Exception(f"Error: Hash mismatch for '{destFile}'. Expected '{expectedHash}' but got '{tmpHash}'")

    xdata = np.zeros([1728, 6])
    ydata = np.zeros([1728])

    mapper = [lambda x : float(x)]*7
    mapper[0] = mapper[1] = lambda x : float([ "vhigh", "high", "med", "low" ].index(x))
    mapper[2] = lambda x : float([ '2', '3', '4', "5more" ].index(x))
    mapper[3] = lambda x : float([ '2', '4', "more" ].index(x))
    mapper[4] = lambda x : float([ "small", "med", "big" ].index(x))
    mapper[5] = lambda x : float([ "low", "med", "high" ].index(x))
    mapper[6] = lambda x : float([ "unacc", "acc", "good", "vgood" ].index(x))

    with open(os.path.join(path, "car.data"), newline="") as f:
        reader = csv.reader(f, delimiter=',')
        
        for row in reader:
            for token, i in zip(row, range(7)):
                if i == 6:
                    ydata[reader.line_num-1] = mapper[i](token)
                else:
                    xdata[reader.line_num-1, i] = mapper[i](token)

    _car_evaluation_xdata = xdata
    _car_evaluation_ydata = ydata
   
    return _car_evaluation_xdata.copy(), _car_evaluation_ydata.copy()

def load_satimage():
    path = _make_data_cache_dir("satimage")

    baseUrl = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/"
    files = [ "sat.trn", "sat.tst", "sat.doc" ]
    hashes = [ "e896dc88a960fa2404160fc4c3cb3dc53fcf4afd80ba920bf2d261bd42d12613", "4b9167b8a92baafafed7c8809aef86d0683a5e685c19d98d119fa6974f0f2479", None ]
    
    for file, expectedHash in zip(files, hashes):
        srcUrl = baseUrl + "/satimage/" + file
        destFile = os.path.join(path, file)
        
        if not os.path.exists(destFile):
            _download_file(srcUrl, destFile)
            
        tmpHash = _hash_file(destFile)
        
        if expectedHash is not None and expectedHash != tmpHash:
            raise Exception(f"Error: Hash mismatch for '{destFile}'. Expected '{expectedHash}' but got '{tmpHash}'")

    trainFile=os.path.join(path, "sat.trn")
    testFile=os.path.join(path, "sat.tst")

    xtrain = np.zeros([4435, 36])
    ytrain = np.zeros([4435])
    
    xtest = np.zeros([2000, 36])
    ytest = np.zeros([2000])

    with open(trainFile, mode="rt", newline="") as f:
        for line, line_num in zip(f, range(4435)):
            values = [ float(x) for x in line.strip().split(" ") ]
            ytrain[line_num] = values[-1]
            xtrain[line_num, :] = values[0:-1]
       

    with open(testFile, mode="rt", newline="") as f:
        for line, line_num in zip(f, range(2000)):
            values = [ float(x) for x in line.strip().split(" ") ]
            ytest[line_num] = values[-1]
            xtest[line_num, :] = values[0:-1]

    # Remove gap in classes (use unused class 6)
    ytrain[ytrain == 7] = 6
    ytest[ytest == 7] = 6

    return xtrain, ytrain-1, xtest, ytest-1

_vehicle_xdata = None
_vehicle_ydata = None

def load_vehicle():
    global _vehicle_xdata, _vehicle_ydata

    if _vehicle_xdata is not None:
        return _vehicle_xdata.copy(), _vehicle_ydata.copy()

    path = _make_data_cache_dir("vehicle")

    baseUrl = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/"

    files = [ f"xa{chr(ord('a') + i)}.dat" for i in range(9) ]
    files.append("vehicle.doc")

    hashes = [ "091ec73ae54031aea8432d00caaef101a445eae13196039db9079b0eac13331d", "3fbc55ac03592425521df18d59ba2ed7ddca626b22f84f5fc3253ebda6b82145", "43536f3aadb2a285fa2ba6e13dacc7013c2298fbd728f011e88669bb44856b8d",
        "371b0de9ba3564d8220d8ad17c5f2f886bafebd2c486de1b0273ceb0c24d3035", "503f9c87feb46dfaab6e3c4275e121640a39c0b30b29f120e679ce8e3fbc7be2", "460e14ec4746a1464d7a03671e09310695c348d6831452cb53172734a6a70531",
        "c279f285c353beaa661168e68feeac726cd446f076f717662635267d89006601", "467735394d1a058e15da81d51fd8e1147d425e7f53392ac6b8e730a34a02049e", "feba2f3aad6db050bc7acbd224cd4e0d32846865ab5cfc53cc59baea2e1ff484",
        None ]

    for file, expectedHash in zip(files, hashes):
        srcUrl = baseUrl + "/vehicle/" + file
        destFile = os.path.join(path, file)
        
        if not os.path.exists(destFile):
            _download_file(srcUrl, destFile)
            
        tmpHash = _hash_file(destFile)
        
        if expectedHash is not None and expectedHash != tmpHash:
            raise Exception(f"Error: Hash mismatch for '{destFile}'. Expected '{expectedHash}' but got '{tmpHash}'")


    xdata = np.zeros([846, 18])
    ydata = np.zeros([846])

    for file, i in zip(files, range(9)):
        offset = 94*i

        with open(os.path.join(path, file), mode="rt", newline="") as f:
            for line, line_num in zip(f, range(94)):
                values = line.strip().split(" ")
                ydata[offset + line_num] = float(["opel", "saab", "bus", "van"].index(values[18]))
                xdata[offset + line_num, :] = [ float(value) for value in values[0:18] ]

    _vehicle_xdata = xdata
    _vehicle_ydata = ydata
        
    return _vehicle_xdata.copy(), _vehicle_ydata.copy()

def load_yeast():
    path = _make_data_cache_dir("yeast")

    baseUrl = "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    files = [ "yeast.data", "yeast.names" ]
    hashes = [ "7cf61776fc04f527f93bf57a327b863893a1225d82df02d457e8950173218258", None ]

    for file, expectedHash in zip(files, hashes):
        srcUrl = baseUrl + "/yeast/" + file
        destFile = os.path.join(path, file)
        
        if not os.path.exists(destFile):
            _download_file(srcUrl, destFile)
            
        tmpHash = _hash_file(destFile)
        
        if expectedHash is not None and expectedHash != tmpHash:
            raise Exception(f"Error: Hash mismatch for '{destFile}'. Expected '{expectedHash}' but got '{tmpHash}'")

    xdata = np.zeros([1484, 8])
    ydata = np.zeros([1484])

    with open(os.path.join(path, "yeast.data"), mode="rt", newline="") as f:
        for line, line_num in zip(f, range(1484)):
            values = [ token for token in line.strip().split(" ") if len(token) > 0 ]
            xdata[line_num, :] = [ float(value) for value in values[1:-1] ] # Skip identifier in column 0
            ydata[line_num] = float(["CYT", "NUC", "MIT", "ME3", "ME2", "ME1", "EXC", "VAC", "POX", "ERL"].index(values[-1]))

    return xdata, ydata

def load_ecoli():
    path = _make_data_cache_dir("ecoli")

    baseUrl = "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    files = [ "ecoli.data", "ecoli.names" ]
    hashes = [ "008bd8fbb1d8b34040c3c8c4e987cac2a7ebf116e008b140cc6441be5261ba1d", None ]

    for file, expectedHash in zip(files, hashes):
        srcUrl = baseUrl + "/ecoli/" + file
        destFile = os.path.join(path, file)
        
        if not os.path.exists(destFile):
            _download_file(srcUrl, destFile)
            
        tmpHash = _hash_file(destFile)
        
        if expectedHash is not None and expectedHash != tmpHash:
            raise Exception(f"Error: Hash mismatch for '{destFile}'. Expected '{expectedHash}' but got '{tmpHash}'")

    xdata = np.zeros([336, 7])
    ydata = np.zeros([336])

    with open(os.path.join(path, "ecoli.data"), mode="rt", newline="") as f:
        for line, line_num in zip(f, range(336)):
            values = [ token for token in line.strip().split(" ") if len(token) > 0 ]
            xdata[line_num, :] = [ float(value) for value in values[1:-1] ] # Skip identifier in column 0
            ydata[line_num] = float(["cp", "im", "pp", "imU", "om", "omL", "imL", "imS"].index(values[-1]))

    return xdata, ydata

# XXX: Has missing data!
def load_dermatology(missing=True):
    path = _make_data_cache_dir("dermatology")

    baseUrl = "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    files = [ "dermatology.data", "dermatology.names" ]
    hashes = [ "455eba77f72cd087ce54a5a266c514c5f34e85000c8689ad09e796d68ad45742", None ]

    for file, expectedHash in zip(files, hashes):
        srcUrl = baseUrl + "/dermatology/" + file
        destFile = os.path.join(path, file)
        
        if not os.path.exists(destFile):
            _download_file(srcUrl, destFile)
            
        tmpHash = _hash_file(destFile)
        
        if expectedHash is not None and expectedHash != tmpHash:
            raise Exception(f"Error: Hash mismatch for '{destFile}'. Expected '{expectedHash}' but got '{tmpHash}'")

    xdata = np.zeros([366, 34])
    ydata = np.zeros([366])

    with open(os.path.join(path, "dermatology.data"), mode="rt", newline="") as f:
        for line, line_num in zip(f, range(366)):
            values = [ token for token in line.strip().split(",") if len(token) > 0 ]
            values = [ (float(value) if value != "?" else np.nan) for value in values ] # Handle missing data
            xdata[line_num, :] = values[0:-1]
            ydata[line_num] = values[-1]

    if not missing:
        ind = ~np.isnan(xdata).any(axis=1) # Remove rows with missing data
        xdata = xdata[ind]
        ydata = ydata[ind]

    return xdata, ydata-1

# XXX: Has missing data!
# Just cleveland
def load_heart_disease(missing=True):
    path = _make_data_cache_dir("heart_disease")

    baseUrl = "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    files = [ "processed.cleveland.data", "heart-disease.names" ]
    hashes = [ "a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8", None ]

    for file, expectedHash in zip(files, hashes):
        srcUrl = baseUrl + "/heart-disease/" + file
        destFile = os.path.join(path, file)
        
        if not os.path.exists(destFile):
            _download_file(srcUrl, destFile)
            
        tmpHash = _hash_file(destFile)
        
        if expectedHash is not None and expectedHash != tmpHash:
            raise Exception(f"Error: Hash mismatch for '{destFile}'. Expected '{expectedHash}' but got '{tmpHash}'")

    xdata = np.zeros([303, 13])
    ydata = np.zeros([303])

    with open(os.path.join(path, "processed.cleveland.data"), mode="rt", newline="") as f:
        for line, line_num in zip(f, range(303)):
            values = [ token for token in line.strip().split(",") if len(token) > 0 ]
            values = [ (float(value) if value != "?" else np.nan) for value in values ] # Handle missing data
            xdata[line_num, :] = values[0:-1]
            ydata[line_num] = 1.0 if values[-1] > 0 else 0.0

    if not missing:
        ind = ~np.isnan(xdata).any(axis=1) # Remove rows with missing data
        xdata = xdata[ind]
        ydata = ydata[ind]

    return xdata, ydata

# XXX: Has missing data!
def load_credit_screening(missing=True):
    path = _make_data_cache_dir("credit_screening")

    baseUrl = "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    files = [ "crx.data", "crx.names" ]
    hashes = [ "fff49bc186cbddb3ace7371d40d9fbbb3af4f126019c13ff3f562249b1454f4d", None ]

    for file, expectedHash in zip(files, hashes):
        srcUrl = baseUrl + "/credit-screening/" + file
        destFile = os.path.join(path, file)
        
        if not os.path.exists(destFile):
            _download_file(srcUrl, destFile)
            
        tmpHash = _hash_file(destFile)
        
        if expectedHash is not None and expectedHash != tmpHash:
            raise Exception(f"Error: Hash mismatch for '{destFile}'. Expected '{expectedHash}' but got '{tmpHash}'")

    mappers = [lambda x : float(x)]*16

    mappers[0] = lambda x : float(['b', 'a'].index(x))
    mappers[3] = lambda x : float(['u', 'y', 'l', 't'].index(x))
    mappers[4] = lambda x : float(['g', 'p', "gg"].index(x))
    mappers[5] = lambda x : float(['c', 'd', "cc", 'i', 'j', 'k', 'm', 'r', 'q', 'w', 'x', 'e', "aa", "ff"].index(x))
    mappers[6] = lambda x : float(['v', 'h', "bb", 'j', 'n', 'z', "dd", "ff", 'o'].index(x))
    mappers[8] = mappers[9] = mappers[11] = lambda x : float(['f', 't'].index(x))
    mappers[12] = lambda x : float(['g', 'p', 's'].index(x))
    mappers[15] = lambda x : float(['-', '+'].index(x))

    xdata = np.zeros([690, 15])
    ydata = np.zeros([690])

    with open(os.path.join(path, "crx.data"), mode="rt", newline="") as f:
        for line, line_num in zip(f, range(690)):
            values = [ token for token in line.strip().split(",") if len(token) > 0 ]
            values = [ (mapper(token) if token != '?' else np.nan) for token, mapper in zip(values, mappers) ]
            xdata[line_num, :] = values[0:-1]
            ydata[line_num] = values[-1]

    if not missing:
        ind = ~np.isnan(xdata).any(axis=1) # Remove rows with missing data
        xdata = xdata[ind]
        ydata = ydata[ind]

    return xdata, ydata

def load_ann_thyroid():
    path = _make_data_cache_dir("thyroid_disease")

    baseUrl = "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    files = [ "ann-train.data", "ann-test.data", "ann-thyroid.names" ]
    hashes = [ "3da53a156bda36cb0c97e9f4b6b111c9226c54c4aa00230de5604b787c47e3a6", "c649ea19416e78c7996cfaaa2a9e281cb597d4b075aaa68c494fc3e4ee3aa30b", None ]

    for file, expectedHash in zip(files, hashes):
        srcUrl = baseUrl + "/thyroid-disease/" + file
        destFile = os.path.join(path, file)
        
        if not os.path.exists(destFile):
            _download_file(srcUrl, destFile)
            
        tmpHash = _hash_file(destFile)
        
        if expectedHash is not None and expectedHash != tmpHash:
            raise Exception(f"Error: Hash mismatch for '{destFile}'. Expected '{expectedHash}' but got '{tmpHash}'")

    xtrain = np.zeros([3772,21])
    ytrain = np.zeros([3772])
    xtest = np.zeros([3428,21])
    ytest = np.zeros([3428])

    with open(os.path.join(path, "ann-train.data"), mode="rt", newline="") as f:
        for line, line_num in zip(f, range(3772)):
            values = [ float(token) for token in line.strip().split(" ") if len(token) > 0 ]
            xtrain[line_num, :] = values[0:-1]
            ytrain[line_num] = values[-1]

    with open(os.path.join(path, "ann-test.data"), mode="rt", newline="") as f:
        for line, line_num in zip(f, range(3428)):
            values = [ float(token) for token in line.strip().split(" ") if len(token) > 0 ]
            xtest[line_num, :] = values[0:-1]
            ytest[line_num] = values[-1]

    return xtrain, ytrain-1, xtest, ytest-1

_optdigits_xtrain = None
_optdigits_ytrain = None
_optdigits_xtest = None
_optdigits_ytest = None

def load_optdigits():
    global _optdigits_xtrain, _optdigits_ytrain, _optdigits_xtest, _optdigits_ytest

    if _optdigits_xtrain is not None:
        return _optdigits_xtrain.copy(), _optdigits_ytrain.copy(), _optdigits_xtest.copy(), _optdigits_ytest.copy()

    path = _make_data_cache_dir("optdigits")

    baseUrl = "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    files = [ "optdigits.tra", "optdigits.tes", "optdigits.names" ]
    hashes = [ "e1b683cc211604fe8fd8c4417e6a69f31380e0c61d4af22e93cc21e9257ffedd", "6ebb3d2fee246a4e99363262ddf8a00a3c41bee6014c373ed9d9216ba7f651b8", None ]

    for file, expectedHash in zip(files, hashes):
        srcUrl = baseUrl + "/optdigits/" + file
        destFile = os.path.join(path, file)
        
        if not os.path.exists(destFile):
            _download_file(srcUrl, destFile)
            
        tmpHash = _hash_file(destFile)
        
        if expectedHash is not None and expectedHash != tmpHash:
            raise Exception(f"Error: Hash mismatch for '{destFile}'. Expected '{expectedHash}' but got '{tmpHash}'")

    xtrain = np.zeros([3823,64])
    ytrain = np.zeros([3823])

    xtest = np.zeros([1797,64])
    ytest = np.zeros([1797])

    with open(os.path.join(path, "optdigits.tra"), mode="rt", newline="") as f:
        for line, line_num in zip(f, range(3823)):
            values = [ float(token) for token in line.strip().split(",") ]
            xtrain[line_num, :] = values[0:-1]
            ytrain[line_num] = values[-1]

    with open(os.path.join(path, "optdigits.tes"), mode="rt", newline="") as f:
        for line, line_num in zip(f, range(1797)):
            values = [ float(token) for token in line.strip().split(",") ]
            xtest[line_num, :] = values[0:-1]
            ytest[line_num] = values[-1]

    _optdigits_xtrain = xtrain
    _optdigits_ytrain = ytrain
    _optdigits_xtest = xtest
    _optdigits_ytest = ytest

    return _optdigits_xtrain.copy(), _optdigits_ytrain.copy(), _optdigits_xtest.copy(), _optdigits_ytest.copy()

_nursery_xdata = None
_nursery_ydata = None

def load_nursery():
    global _nursery_xdata, _nursery_ydata

    if _nursery_xdata is not None:
        return _nursery_xdata.copy(), _nursery_ydata.copy()

    path = _make_data_cache_dir("nursery")

    baseUrl = "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    files = [ "nursery.data", "nursery.names" ]
    hashes = [ "8e0389c3dd37590248a921c2726d869ee96b817761a35eb8416afa24f31f931d", None ]

    for file, expectedHash in zip(files, hashes):
        srcUrl = baseUrl + "/nursery/" + file
        destFile = os.path.join(path, file)
        
        if not os.path.exists(destFile):
            _download_file(srcUrl, destFile)
            
        tmpHash = _hash_file(destFile)
        
        if expectedHash is not None and expectedHash != tmpHash:
            raise Exception(f"Error: Hash mismatch for '{destFile}'. Expected '{expectedHash}' but got '{tmpHash}'")

    xdata = np.zeros([12960,8])
    ydata = np.zeros([12960])

    mappers = [lambda x : float(x)]*9
    mappers[0] = lambda x : float(["usual", "pretentious", "great_pret"].index(x))
    mappers[1] = lambda x : float(["proper", "less_proper", "improper", "critical", "very_crit"].index(x))
    mappers[2] = lambda x : float(["complete", "completed", "incomplete", "foster"].index(x))
    mappers[3] = lambda x : float(['1', '2', '3', "more"].index(x))
    mappers[4] = lambda x : float(["convenient", "less_conv", "critical"].index(x))
    mappers[5] = lambda x : float(["convenient", "inconv"].index(x))
    mappers[6] = lambda x : float(["nonprob", "slightly_prob", "problematic"].index(x))
    mappers[7] = lambda x : float(["recommended", "priority", "not_recom"].index(x))
    mappers[8] = lambda x : float(["not_recom", "recommend", "very_recom", "priority", "spec_prior"].index(x))

    with open(os.path.join(path, "nursery.data"), mode="rt", newline="") as f:
        for line, line_num in zip(f, range(12960)):
            values = [ mapper(token) for token, mapper in zip(line.strip().split(","),mappers) ]
            xdata[line_num, :] = values[0:-1]
            ydata[line_num] = values[-1]

    _nursery_xdata = xdata
    _nursery_ydata = ydata

    return _nursery_xdata.copy(), _nursery_ydata.copy()

datasets["abalone"] = load_abalone
datasets["abalone_reg"] = load_abalone_reg
datasets["iris"] = load_iris
datasets["mnist"] = load_mnist
datasets["poker"] = load_poker
datasets["letter"] = load_letter
datasets["har"] = load_har
datasets["madelon"] = load_madelon
datasets["cifar10"] = load_cifar10
datasets["usps"] = load_usps
datasets["forest_cover"] = load_forest_cover
datasets["breast_cancer"] = load_breast_cancer
datasets["pima"] = load_pima
datasets["car_evaluation"] = load_car_evaluation
datasets["satimage"] = load_satimage
datasets["vehicle"] = load_vehicle
datasets["yeast"] = load_yeast
datasets["ecoli"] = load_ecoli
datasets["dermatology"] = load_dermatology
datasets["heart_disease"] = load_heart_disease
datasets["credit_screening"] = load_credit_screening 
datasets["ann_thyroid"] = load_ann_thyroid
datasets["load_optdigits"] = load_optdigits
datasets["nursery"] = load_nursery

if __name__ == "__main__":
    pass

