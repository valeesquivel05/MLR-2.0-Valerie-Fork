Requirements to install:

conda install

conda create --name yourname37 python=3.7
conda activate yourname37
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
pip install scikit-learn
conda install matplotlib
conda install tqdm imageio ipython opencv pandas

To train a new model:
 - Open Training.py, on line 15, set the 'data_set_flag' variable to one of the following data sets.
 - Data set options: 'mnist', 'cifar10', 'padded_mnist', 'padded_cifar10', and 'padded_mnist_rg'.
 - A padded dataset consists of images placed within a 28 * 100 retina.
 - Save and run Training.py, the model's checkpoints will be saved to a folder titled output/ by default.
 
To utilize an already trained model:
 - Import the load_checkpoint function from mVAE.py and call it with the model's checkpoint filename:
	   from mVAE import load_checkpoint
	   load_checkpoint('CHECKPOINTFILENAME.pth')
 
 - Note: the dimensions of the loaded checkpoints must match the dimensions of the VAE_CNN defined in mVAE.py on lines 179 - 183.

Interacting with the model:
 - All functions for interacting with the model are in mVAE.py.
 - dataset_builder(data_set_flag, batch_size, return_class = False)
 	.
 - VAE_CNN class:
   - forward pass: vae(data, whichdecode, keepgrad)
   	.
   - encoder: 
   - decoders:
    
 - progress_out(data, epoch, count, skip, filename)
