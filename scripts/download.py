import urllib.request
import tarfile

url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
filename = "cifar-10-python.tar.gz"

urllib.request.urlretrieve(url, filename)

with tarfile.open(filename, "r:gz") as tar:
    tar.extractall("./data")