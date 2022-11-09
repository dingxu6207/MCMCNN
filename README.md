# MCMC_NN

filefold is the model trained by corresponding filters

conda install -c conda-forge emcee
pip install tensorflow
python -m pip install corner

Three areas need to be modified when there is a new target：

the first area is the filename
fileone = 'KIC 6431545.txt'

the second area: if there is no third light effect, index = 0, the range of l3 is comment.

index = 0 
#initial space[T/5850，incl/90,q,f,t2t1,l3,offset1, offset2]
init_dist = [(5459/5850-0.0001, 5459/5850+0.0001), 
             (54.40/90-6/90, 54.40/90+10/90), 
             (0.5, 3), 
             (0.05, 0.9), 
             (0.8, 1.2),
             #(0, 1),
             (-10,10),
             (-0.01,0.01)
             ]
 
the third area: if there is third light effect, index = 1, the range of l3 is uncomment.
index = 1 
#initial space[T/5850，incl/90,q,f,t2t1,l3,offset1, offset2]
init_dist = [(5459/5850-0.0001, 5459/5850+0.0001), 
             (54.40/90-6/90, 54.40/90+10/90), 
             (0.5, 3), 
             (0.05, 0.9), 
             (0.8, 1.2),
             (0, 1),
             (-10,10),
             (-0.01,0.01)
             ]
