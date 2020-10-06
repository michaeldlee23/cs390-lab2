# CS390-NIP Lab 2: Convolutional Neural Networks
---
Run `python lab2.py -h` to see usage instructions
Example: `python lab2.py -a tf_conv -d mnist_f -e 20`

### Viewing Learning Curves via Tensorboard
Run `tensorboard --logdir logs/fit` to startup local tensorboard, reachable at localhost:6006 once started
* Recorded logs are stored in `/logs/old`
* Logs are grouped in "rounds"
  * A round consists of saved logs for all five datasets, run on the same architecture of network
* To view a log, move the desired folder from `/logs/old/round/` to `/logs/fit/`, then refresh tensorboard
  * May need to create /logs/fit/ folder
  * To view more than one dataset at a time, move all desired logs into `/logs/fit/`

### Viewing Saved Models
Run `python lab2.py -l path-to-saved-model`
* Saved models are stored in `/models`
* Models are grouped in "rounds"
  * A round consists of saved models for all five datasets, run on the same architecture of network
* Each saved model has a `meta.txt` file which consists of a quick summary of hyperparameters and accuracy

Example: `python lab2.py -l ./models/round3/tf_conv-cifar_100_f-2020-10-05-22.10.25`

