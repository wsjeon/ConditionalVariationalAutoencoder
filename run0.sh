#!/bin/bash
export CUDA_VISIBIE_DEVICES="0" && python main.py --LR 0.01 &
sleep 15
export CUDA_VISIBIE_DEVICES="1" && python main.py --LR 0.005 &
sleep 15
#export CUDA_VISIBIE_DEVICES="2" && python main.py --LR 0.0025 &
#sleep 15
#export CUDA_VISIBIE_DEVICES="3" && python main.py --LR 0.001 &
#sleep 15
#export CUDA_VISIBIE_DEVICES="4" && python main.py --LR 0.0005 &
#sleep 15
#export CUDA_VISIBIE_DEVICES="5" && python main.py --LR 0.00025 &
#sleep 15
#export CUDA_VISIBIE_DEVICES="6" && python main.py --LR 0.0001 &
#sleep 15
#export CUDA_VISIBIE_DEVICES="7" && python main.py --LR 0.00005 &
