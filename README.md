## CS274E Final Project

In this project, we implement the Cycle-Consistent Adversarial Networks proposed by [Jun-Yan et al](https://junyanz.github.io/CycleGAN/).

### Get Started

Before running this program, you need to download the dataset.
```bash
$ bash ./datasets/download_dataset.sh horse2zebra
```

Then, run the following commands, and the program will output the sample pictures in the `outputs` folder. 
```bash
$ pip install -r requirements.txt
$ python3 train.py
```