# CS390-NIP Lab 2: Convolutional Neural Networks
---
Run `python lab2.py -h` to see usage instructions

### Viewing Learning Curves via Tensorboard
Run `tensorboard logdir logs/fit` to startup local tensorboard, reachable at localhost:6006 once started
* Recorded logs are stored in `/logs/old`
* To view a log, move the desired folder from `/logs/old` to `/logs/fit`, then refresh tensorboard

### Viewing Saved Models
Run `python lab2.py -l path-to-saved-model`
* Saved models are stored in `/models`
* Models are grouped in "rounds"
  * A round consists of saved models for all five datasets, run on the same architecture of CNN
* Each saved model has a `meta.txt` file which consists of a quick summary of hyperparameters and accuracy
