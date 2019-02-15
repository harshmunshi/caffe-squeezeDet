# caffe-squeezeDet

#### Model Conversion Note (Harsh Munshi)
```
$ pip install tensorpack
```
Place your tensorflow models to tf_models under tfutils. Then run,

```
$ python dump_to_npz.py --meta ./tf_models/<some_model>.ckpt-xxxx.meta ./tf_models/<some_model>.ckpt-xxxx out.npz
```

#### Support for npy files
Place your tensorflow models to tf_models under tfutils. Then run,

```
$ python dump_to_npy.py --meta ./tf_models/<some_model>.ckpt-xxxx.meta ./tf_models/<some_model>.ckpt-xxxx 
```

#### **UPDATE

You can also do it without tensorpack, use the following code:

```
from tensorflow.python.tools import inspect_checkpoint as chkp

chkp.print_tensors_in_checkpoint_file("./model	/model.ckpt-12000", tensor_name='', all_tensors=True)
```

#### Conversion to caffemodel

Once you have the .npy or .npz files, copy them and paste it to model_checkpoints folder. Then run

```
$ python npy2caffe.py
```
Now you have the caffemodel :)

#### This is the caffe version of squeezeDet. And I converted tensorflow  model directly into caffemodel. 
----
### Note
----
The convolution operation is different in tensorflow and caffe, especially the padding iterm. Thus, using directly converted model will cause problems. **Trick**:  using a pad = 2 convolution operation for CONV1 Layer to get a larger feature map and than crop into the right size. 

But for kernel size 3 x 3 or 1 x 1 and stride step 1 with padding SAME in tensorflow, we directly choose convolution param with pad = 1, kernel size = 3 and stride = 1

Additional note (Harsh Munshi): Direct cropping to KITTI also works.
### Demo
``` python
cd caffe-squeezeDet
python ./src/demo.py
```
