sudo apt-get install git vim
sudo apt-get install python-pip python-dev python-virtualenv python-pip python-tk
sudo apt-get install python-opencv
pip install moviepy h5py matplotlib easydict numpy lmdb scikit-image

virtualenv --system-site-packages ~/tensorflow11
source ~/tensorflow11/bin/activate
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.11.0-cp27-none-linux_x86_64.whl
pip install --upgrade $TF_BINARY_URL
