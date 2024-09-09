import math
import time
import sys
import numpy as np
import random

import matplotlib
from matplotlib import pyplot as plt

# import miyamoto_nagai potential from galpy
from galpy.potential import MiyamotoNagaiPotential

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
#device = torch.device('mps')
print("INFO: current device: {}".format(device))

import nflows
from nflows import flows
from nflows import transforms
from nflows import distributions

print('Modules loaded :)')

print(f'\nINITIALISATION VARIANCE CODE :)\n')



def janky_stream(n,R,phimax,v,seed=0, rstd=0.01, thetastd=0.01, vstd=0.01):
    np.random.seed(seed)

    bodies = np.zeros((n, 7))  # Each row: mass, x, y, z, vx, vy, vz
    bodies[:, 0] = 1.0 / n

    ## Generate positions
    # select phi between 0 and phimax
    phi = np.random.uniform(0, phimax, n)
    # select R as a gaussian with mean R and std of rstd, same for theta
    R = np.random.normal(R, rstd, n)
    theta = np.random.normal(np.pi/2., thetastd, n)

    bodies[:, 1] = R * np.sin(theta) * np.cos(phi)
    bodies[:, 2] = R * np.sin(theta) * np.sin(phi)
    bodies[:, 3] = R * np.cos(theta)

    ## generate velocities
    # select v as a gaussian with mean v and std of 0.01
    v = np.random.normal(v, vstd, n)
    # all the direction should be the same and circular
    bodies[:, 4] = v * np.sin(theta) * np.sin(phi)
    bodies[:, 5] = v * np.sin(theta) * np.cos(phi)
    bodies[:, 6] = v * np.cos(theta)

    return bodies

def disk_and_stream(disk_bodies, fraction_stream, R=1.5, phimax=np.pi/4, seed=0, rstd=0.01, thetastd=0.01, vstd=0.01):
    n = len(disk_bodies)
    n_stream = int(n*fraction_stream)
    n_disk = n - n_stream

    bodies = disk_bodies[:n_disk]
    # the velocity of the stream is the circular velocity at R
    # initialise the Miyaamoto-Nagai potential
    MN = MiyamotoNagaiPotential(a=1.0, b=0.1, normalize=1)
    v = MN.vcirc(R)
    #v = np.sqrt((1.-fraction_stream)*(np.sqrt( R**2 * (R**2 + 1.)**(-3./2.) )))
    print(v)
    bodies_stream = janky_stream(n_stream, R,phimax,v,seed=seed, rstd=rstd, thetastd=thetastd, vstd=vstd)
    return np.vstack((bodies, bodies_stream))

## Loading in the plummer sphere data
#data = np.load('plummer_dumb.npy').astype(np.float32)
#data = np.load('plummer_dumbish.npy').astype(np.float32)
#print(f'data shape: {data.shape}')

## Some data preprocessing
def remove_rows_exceeding_threshold(array, threshold):
    mask = np.any(array > threshold, axis=1)
    filtered_array = array[~mask]
    return filtered_array

def remove_rows_under_threshold(array, threshold):
    mask = np.any(array < threshold, axis=1)
    filtered_array = array[~mask]
    return filtered_array

threshold = 30
#data = remove_rows_exceeding_threshold(data, threshold)
#data = remove_rows_under_threshold(data, -threshold)

## Define the initial MAF and load it in every time before beginning training

#define MAF
# hyper-parameters
INPUT_DIM       = 6
NUM_MAFS        = 6
HIDDEN_FEATURES = 32
NUM_BLOCKS      = 2
PERMUTATION     = [5,0,1,2,3,4]   # must be a permutation of INPUT_DIM integers.
#PERMUTATION     = [5,0,1,2,3,4]   # must be a permutation of INPUT_DIM integers.

# Set random seeds for reproducibility
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

## flow definitions
base_dist = nflows.distributions.normal.StandardNormal(shape=[INPUT_DIM])

list_transforms = []
for k in range(NUM_MAFS):
    if k != 0:
        list_transforms.append(
            nflows.transforms.permutations.Permutation(torch.tensor(PERMUTATION))
        )
    list_transforms.append(
        nflows.transforms.autoregressive.MaskedAffineAutoregressiveTransform(
            features=INPUT_DIM,
            hidden_features=HIDDEN_FEATURES,
            num_blocks=NUM_BLOCKS,
            activation=torch.nn.functional.gelu
        )
    )

transform = nflows.transforms.base.CompositeTransform(list_transforms).to(device)

flow = nflows.flows.base.Flow(transform, base_dist).to(device)


## MAF initialisation variance

n_flow = 50

#train_sizes = [10000,25000,50000,75000,150000,200000]

# Assuming the input is in the form "python train.py [1000,2000,3000] 1.0e-2"
arg = sys.argv[1]
train_sizes_str = arg.strip("[]")  # Remove the brackets
train_sizes = list(map(int, train_sizes_str.split(',')))  # Split by comma and convert to integers
print(f"Train Sizes: {train_sizes}")

# loading in the target stream fraction
arg = sys.argv[2]
train_sizes_str = arg.strip("[]")  # Remove the brackets
fraction_stream_str = arg
fraction_stream = float(arg)
print(f"Stream Fraction: {fraction_stream}")

# use the same n_flow and train_size as above
losses_init = np.empty(n_flow, dtype=object)
init_seeds = np.arange(42,42+n_flow,1)

#original_stdout = sys.stdout

print('ready to start process...')

# Start timing
start_time = time.perf_counter()

#preprocessing
#arr_pos=remove_rows_exceeding_threshold(arr_pos,30)
#arr_pos=remove_rows_under_threshold(arr_pos,-30)
#print(data.shape,flush=True)
#scale = np.sqrt((data**2).mean(axis=0))  # mean=0 is assumed
#print(f"scale: {scale}")
#arr_scale = np.array(scale)
#np.save(preprocessor_values[i], arr_scale)

def preprocess(arr_input):
    return arr_input / scale
def preprocess_inv(arr_input):
    return arr_input * scale

#print('\npreprocessing the data...')
#arr_pos_prep = preprocess(data)
#print('\npreprocessing done')

# Load in all the data files
total_array = np.load('./MN_bodies.npy')[:,:].astype(np.float32)
# remove the first 1000000 rows as the validation set
arr_pos_prep_valid = total_array[0:1000000,:].astype(np.float32)
arr_pos_prep = total_array[1000000:,:].astype(np.float32)

#preprocessing functions
def preprocess_scale(arr_input,scale):
    return arr_input / scale
def preprocess_inv_scale(arr_input,scale):
    return arr_input * scale


## COMMENT THIS OUT IF YOU WANT TO USE ORIGINAL DATA SLICES
#arr_pos_prep_valid = plummer_and_stream(1000000, fraction_stream, seed=0)[:, 1:].astype(np.float32)
arr_pos_prep_valid = disk_and_stream(arr_pos_prep_valid, fraction_stream)[:, 1:].astype(np.float32)
arr_pos_prep_valid = remove_rows_exceeding_threshold(arr_pos_prep_valid, threshold)
arr_pos_prep_valid = remove_rows_under_threshold(arr_pos_prep_valid, -threshold)
arr_pos_prep_valid -= np.mean(arr_pos_prep_valid,axis=0) # subtract the mean
scale_valid = np.std(arr_pos_prep_valid,axis=0)  # mean=0 is assumed
arr_scale_valid = np.array(scale_valid)
print(f"scale_valid: {scale_valid}")
arr_pos_prep_valid = preprocess_scale(arr_pos_prep_valid,scale_valid)


# Create a file and redirect the standard output to it
for train_size in train_sizes:
    
    #losses_init = np.empty(n_flow, dtype=object)
    losses_init = []
    
    #sys.stdout = file

    for i in range (0,n_flow):

        print("\n Working on flow "+str(i),flush=True)

        #Prepare train/valid dataset split

        # split dataset -> train:valid = 1:1
        
        ## COMMENT THIS OUT IF YOU WANT TO USE THE ORIGINAL DATA TO DERIVE THE SCALE
        if i == 0:
            #arr_pos = mkplummer_vectorised(train_size, seed=42)[:, 1:].astype(np.float32)
            #arr_pos = plummer_and_stream(train_size, fraction_stream, seed=42)[:, 1:].astype(np.float32)
            arr_pos = arr_pos_prep[0:train_size]
            arr_pos = disk_and_stream(arr_pos, fraction_stream)[:, 1:].astype(np.float32)
            arr_pos = remove_rows_exceeding_threshold(arr_pos, threshold)
            arr_pos = remove_rows_under_threshold(arr_pos, -threshold)
            arr_pos -= np.mean(arr_pos,axis=0) # subtract the mean
            scale = np.std(arr_pos,axis=0)  # mean=0 is assumed
            print(f"scale: {scale}")
            arr_scale = np.array(scale)

            #arr_pos_prep_valid = preprocess_scale(arr_pos_prep_valid,scale_valid)
            arr_pos_prep_train = preprocess_scale(arr_pos,scale)

        ## UN-COMMENT THIS IF YOU WANT TO SLICE ORIGINAL DATA
        #arr_pos_prep_train = arr_pos_prep[0:train_size] # initialise training data to be first N samples, but could be any N
        #arr_pos_prep_valid = arr_pos_prep[-train_size:]
        print("training dataset   | shape: ", arr_pos_prep_train.shape,flush=True)
        print("max  : ",arr_pos_prep_train.max(axis=0),flush=True)
        print("min  : ",arr_pos_prep_train.min(axis=0),flush=True)
        print("mean : ",arr_pos_prep_train.mean(axis=0),flush=True)
        print("std  : ",arr_pos_prep_train.std(axis=0),flush=True)
        print("validation dataset | shape: ", arr_pos_prep_valid.shape,flush=True)
        print("max  : ",arr_pos_prep_valid.max(axis=0),flush=True)
        print("min  : ",arr_pos_prep_valid.min(axis=0),flush=True)
        print("mean : ",arr_pos_prep_valid.mean(axis=0),flush=True)
        print("std  : ",arr_pos_prep_valid.std(axis=0),flush=True)

        #define MAF

        # hyper-parameters
        INPUT_DIM       = 6
        NUM_MAFS        = 6
        HIDDEN_FEATURES = 32
        NUM_BLOCKS      = 2
        PERMUTATION     = [5,0,1,2,3,4]   # must be a permutation of INPUT_DIM integers.
        #PERMUTATION     = [5,0,1,2,3,4]   # must be a permutation of INPUT_DIM integers.

        ## flow definitions
        base_dist = nflows.distributions.normal.StandardNormal(shape=[INPUT_DIM])

        list_transforms = []
        for k in range(NUM_MAFS):
            if k != 0:
                list_transforms.append(
                    nflows.transforms.permutations.Permutation(torch.tensor(PERMUTATION))
                )
            list_transforms.append(
                nflows.transforms.autoregressive.MaskedAffineAutoregressiveTransform(
                    features=INPUT_DIM,
                    hidden_features=HIDDEN_FEATURES,
                    num_blocks=NUM_BLOCKS,
                    activation=torch.nn.functional.gelu
                )
            )

        # initialising the random MAF with the corresponding random seed
        torch.manual_seed(init_seeds[i])

        transform = nflows.transforms.base.CompositeTransform(list_transforms).to(device)

        flow = nflows.flows.base.Flow(transform, base_dist).to(device)

        num_param = sum(p.numel() for p in flow.parameters())
        if i==0:print("number of params: ", num_param,flush=True)

        #training

        # we will use ADAM with default parameters:
        LEARNING_RATE = 0.001

        # define optimizer: 
        optimizer = torch.optim.Adam(
            flow.parameters(),
            lr=LEARNING_RATE
        )

        # negative log likelihood loss function 
        #   mode: mean or sum
        def calc_loss(tensor_input, mode="mean"):
            arr_nll = -flow.log_prob(tensor_input)
            if mode == "mean":
                return arr_nll.mean()
            elif mode == "sum":
                return arr_nll.sum()
            else:
                assert False

        # training parameters
        ##############################
        num_epoch_max = 1000
        batch_size_train     = 2**10
        batch_size_inference = 2**15

        # parameters for early stopping
        patience_max=50

        # temporary file save path
        path_temp_wgt = "./wgt_maf_"+str(train_size)+str(fraction_stream)+"_init"#path_temp_wgt = path_temp_wgts[i]
        #path_temp_wgt = "./wgt_maf_"+str(train_size)+"_init"#path_temp_wgt = path_temp_wgts[i]
        path_log_file = "./wgt_maf_"+str(train_size)+str(fraction_stream)+"_init.log"#path_log_file = path_log_files[i]
        #path_log_file = "./wgt_maf_"+str(train_size)+"_init.log"#path_log_file = path_log_files[i]

        #training
        # auxilary function for logging purpose
        def print_log(fp, txt):
            print(txt,flush=True)
            fp.writelines(txt+"\n")
            fp.flush()

        # training parameters
        ##############################
        n_batch_train = arr_pos_prep_train.shape[0] // batch_size_train + 1
        n_batch_train_inf = arr_pos_prep_train.shape[0] // batch_size_inference + 1
        n_batch_valid_inf = arr_pos_prep_valid.shape[0] // batch_size_inference + 1

        # timer...
        time_start = time.time()

        # best loss value
        loss_best = np.inf

        # early stopping: patience
        patience=0  

        # load data to GPU:
        tensor_pos_prep_train = torch.tensor(arr_pos_prep_train, device=device)
        tensor_pos_prep_valid = torch.tensor(arr_pos_prep_valid, device=device)

        # split data into batches for loss evaluation
        list_tensor_pos_prep_train = torch.tensor_split(tensor_pos_prep_train, n_batch_train_inf)
        list_tensor_pos_prep_valid = torch.tensor_split(tensor_pos_prep_valid, n_batch_valid_inf)

        # monitor value
        list_loss_train = []
        list_loss_valid = []

        # training loop!
        ###########################
        # open log file
        with open(path_log_file, 'w') as fp:

            # training loop:
            for epoch in range(num_epoch_max):

                # calculate training loss 
                loss_train = torch.tensor(0., dtype=torch.float32, device=device)
                with torch.inference_mode():
                    for buf_pos_prep in list_tensor_pos_prep_train:
                        loss_train = loss_train + calc_loss(buf_pos_prep, "sum")
                    loss_train = loss_train.detach().cpu().numpy() / tensor_pos_prep_train.shape[0]
                    list_loss_train.append(loss_train)

                # calculate validation loss
                loss_valid = torch.tensor(0., dtype=torch.float32, device=device)
                with torch.inference_mode():
                    for buf_pos_prep in list_tensor_pos_prep_valid:
                        loss_valid = loss_valid + calc_loss(buf_pos_prep, "sum")
                    loss_valid = loss_valid.detach().cpu().numpy() / tensor_pos_prep_valid.shape[0]
                    list_loss_valid.append(loss_valid)

                # is model improved?
                if loss_best > loss_valid: 
                    #print_log(fp, "epoch {:04d}: loss improved {:.4f} -> {:.4f}".format(epoch, loss_best, loss_valid))
                    loss_best = loss_valid
                    torch.save(flow.state_dict(), path_temp_wgt)
                    patience=0
                else:
                    # if model is not improved, check the early stopping criterion
                    if patience < patience_max:
                        patience += 1
                    else:
                        print_log(fp, "maximum patience reached")
                        break

                # shuffle and generate minibatches for training
                batches_tensor_pos_prep_train = torch.tensor_split(
                    tensor_pos_prep_train[
                        torch.randperm(tensor_pos_prep_train.shape[0], device=device)
                    ], 
                    n_batch_train
                )

                # update model!
                for buf_pos_prep in batches_tensor_pos_prep_train:
                    # zero grad
                    optimizer.zero_grad()

                    # evaluate training loss and gradients
                    buf_loss = calc_loss(buf_pos_prep, "mean")
                    buf_loss.backward()

                    # update model    
                    optimizer.step()

                # print timer information....
                if (epoch+1)<5 or (epoch+1) % 50 == 0:
                    time_current = time.time()
                    print_log(fp, "INFO: epoch {:04d} | time_elapsed: {:.3f}s".format(epoch+1, time_current - time_start))
                    print_log(fp, "INFO:            | loss_train: {:.4f} | loss_valid: {:.4f} | loss_best: {:.4f}".format(loss_train, loss_valid, loss_best))


            # training loop finished
            #losses_init[i]=loss_best
            losses_init.append(loss_best)
            print("validation loss: "+str(loss_best),flush=True)
            
            flow.load_state_dict(torch.load(path_temp_wgt, map_location='cpu'))
            flow.to(device)
            #plt.plot(plot_flow(flow)[0],plot_flow(flow)[1],c='grey',alpha=0.25)

            print_log(fp, "INFO: training finished")
            np.save(str(train_size)+'_'+str(fraction_stream)+'_initvariance_losses.npy',losses_init)
            #np.save(str(train_size)+'_initvariance_losses.npy',losses_init)

            '''block_vars = ['transform','base_dist','list_transforms','flow','scale', 'arr_scale', 'arr_pos_prep', 'arr_pos_prep_train', 'arr_pos_prep_valid', 'num_param',
                  'optimizer', 'calc_loss', 'num_epoch_max', 'batch_size_train', 'batch_size_inference',
                  'patience_max', 'path_temp_wgt', 'path_log_file', 'n_batch_train', 'n_batch_train_inf',
                  'n_batch_valid_inf', 'time_start', 'loss_best', 'patience', 'tensor_pos_prep_train',
                  'tensor_pos_prep_valid', 'list_tensor_pos_prep_train', 'list_tensor_pos_prep_valid',
                  'list_loss_train', 'list_loss_valid']

            for var in block_vars:
                if var in locals():
                    del locals()[var]'''

    '''plt.plot(plot_true()[0],plot_true()[1],c='black',label='true')
    plt.xlabel('x')
    plt.ylabel('f')
    plt.title(str(train_size)+' samples (data variance)')
    plt.text(-2,0.023,'mean KL: '+str(round(np.mean(losses),5))+'\n std KL: '+str(round(np.std(losses),5)))
    plt.legend()
    #plt.ioff()
    plt.show()'''

    #np.save(str(train_size)+'_initvariance_losses_gentest.npy',losses_init)

# End timing
end_time = time.perf_counter()

# Calculate the duration
duration = end_time - start_time
print(f"\nThe code took {duration} seconds to run.")


