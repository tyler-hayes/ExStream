# ExStream
This is a PyTorch implementation of the ExStream method from our paper "Memory Efficient Experience Replay for Streaming Learning." (https://arxiv.org/abs/1809.05922) 

If using this code, please cite: T.L. Hayes, N.D. Cahill, and C. Kanan. Memory efficient experience replay in streaming learning. In: Proc. IEEE International Conference on Robotics and Automation (ICRA), 2019

Run the main function of 'run_exstream_experiment.py' to generate the results for the CUB-200 experiment with capacity sizes [2,4,8,16] for the 'iid' and 'class_iid' data orderings. 

After each experiment type and capacity has been run, run "plot_results.py' to make plots of each of the experimental results.

This implementation was tested with Python 3.5 and PyTorch 1.0.0.

Note: Our original paper used class-specific buffers for storing exemplars. In this implementation, you can maintain class-specific buffers by setting the buffer_type parameter to 'class'; we also give the option to maintain a single buffer that will fill to full capacity and then begin replacement/merging by setting the buffer_type parameter to 'single'
