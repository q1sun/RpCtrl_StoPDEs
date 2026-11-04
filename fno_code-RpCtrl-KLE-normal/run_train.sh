#!/bin/bash

python3 train_KLE_normal.py --checkpoint Checkpoints/Model_RobPathCtrl_KLE_Normal/simulation_1 --output Checkpoints/RobPathCtrl_KLE_Normal/simulation_1

python3 train_KLE_normal.py --checkpoint Checkpoints/Model_RobPathCtrl_KLE_Normal/simulation_2 --output Checkpoints/RobPathCtrl_KLE_Normal/simulation_2

python3 train_KLE_normal.py --checkpoint Checkpoints/Model_RobPathCtrl_KLE_Normal/simulation_3 --output Checkpoints/RobPathCtrl_KLE_Normal/simulation_3

python3 train_KLE_normal.py --checkpoint Checkpoints/Model_RobPathCtrl_KLE_Normal/simulation_4 --output Checkpoints/RobPathCtrl_KLE_Normal/simulation_4

python3 train_KLE_normal.py --checkpoint Checkpoints/Model_RobPathCtrl_KLE_Normal/simulation_5 --output Checkpoints/RobPathCtrl_KLE_Normal/simulation_5

python3 train_KLE_normal.py --checkpoint Checkpoints/Model_RobPathCtrl_KLE_Normal/simulation_6 --output Checkpoints/RobPathCtrl_KLE_Normal/simulation_6

module add matlab
matlab -nodesktop -nosplash -nodisplay -r "try, train_KLE_normal; catch e, disp(e); end, exit"