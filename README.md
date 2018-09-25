# tensorflow-mnist
Tensorflow MNIST and preprocessing

This project have been forked from the excellent project tensorflow-mnist by Ole Kröger:
 https://medium.com/@o.kroeger/tensorflow-mnist-and-your-own-handwritten-digits-4d1cd32bbab4
 https://github.com/opensourcesblog/tensorflow-mnist
I appreciate all the work he has done for it and I want to acknowledge it.

This is a project familiarizing myself with tensorflow and the MNIST database.
The aim is to develop some kind of handwriting software.
I will try to add some kind of letter/digit recognition to be able to extract the handwritten words and letters/digits from a photo.

For me to be able to run this software, there were some steps I had to take:
(This is from my installation notes)
Install 64-bit Python 3.5 (minimum):
 Python35 64-bit
  https://www.python.org/ftp/python/3.5.4/python-3.5.4-amd64-webinstall.exe
   install for all users
   Python35 to PATH
   Install debug symbols
   
Install tensorflow, pandas and numpy:
 In a command window (do it anywhere after Python35 is in PATH):
  pip3 install --update tensorflow
  pip install pandas
  pip install numpy
  
<Already done in this branch>
 Convert GIT-files from Python2 into Python3:
  CommandPrompt:
  > cd C:\GIT\github\tensorflow-mnist.git\
  > python C:\Program Files\Python35\Tools\scripts\2to3.py -w mnist.py
  > python C:\Program Files\Python35\Tools\scripts\2to3.py -w step2.py
  > python C:\Program Files\Python35\Tools\scripts\2to3.py -w input_data.py
  > python C:\Program Files\Python35\Tools\scripts\2to3.py -w learn_extra.py
</Already done in this branch>

Install OpenCV:
 https://solarianprogrammer.com/2016/09/17/install-opencv-3-with-python-3-on-windows/
  https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads
   vc_redist.x64.exe
  https://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv
   opencv_python-3.4.3-cp35-cp35m-win_amd64.whl
   pip install "opencv_python-3.4.3-cp35-cp35m-win_amd64.whl"
   
 There was an error:
  "TypeError: only integer scalar arrays can be converted to a scalar index"
  LINE33 bytestream.read
  https://stackoverflow.com/questions/42128830/typeerror-only-integer-scalar-arrays-can-be-converted-to-a-scalar-index
  Change:
   def _read32(bytestream):
    dt = numpy.dtype(numpy.uint32).newbyteorder('>')
    return numpy.frombuffer(bytestream.read(4), dtype=dt)
    To:
   def _read32(bytestream):
   dt = numpy.dtype(numpy.uint32).newbyteorder('>')
   return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]
   
 Don't mind the warning:
  2018-09-19 09:45:54.456223: I T:\src\github\tensorflow\tensorflow\core\platform\cpu_feature_guard.cc:141]
   Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
   https://stackoverflow.com/questions/47068709/your-cpu-supports-instructions-that-this-tensorflow-binary-was-not-compiled-to-u
   
SUCCESS will be in the form of something like this:
0.9145
[8 0 4 3]
1.0
