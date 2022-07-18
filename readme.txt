pip install numpy
pip install opencv-python
pip install scikit-learn
pip install imutils

conda install -c conda-forge dlib

# Win 10
conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge

# How to run
python main.py <image_path>
python main.py "images/Hillary.jpg"