#!/bin/bash

for i in {1..5}
do 
    poetry run python -m project.main --config-name=mnist use_wandb=true strategy=fedavg fed="mnist_$i" task=mnist_low_lr
done

for i in {1..5}
do
    poetry run python -m project.main --config-name=mnist use_wandb=true strategy=afl fed="mnist_$i" task=mnist_low_lr
done


for i in {1..5}
do
    poetry run python -m project.main --config-name=mnist use_wandb=true strategy=afl fed="mnist_$i"
done

