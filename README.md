# Revenue
Training and prediction for revenue using a RNN in Keras / Tensorflow.

## There are 2 python scripts:
Both make company data from whats avaiable in the snowflake database and:
  1. Train RNNs or 
  2. produce necessary predctions using the trained RNNs
  
The models are saved out via json & hdf5 (.h5) files into s3 buckets in AWS. They are then re-loaded back into the enviroment to make weekly predictions.
