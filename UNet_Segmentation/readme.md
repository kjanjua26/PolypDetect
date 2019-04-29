We train a UNET on two different POLYP datasets to perform mask segmentation.
For datasets, we used the KVASIR dataset: https://datasets.simula.no/kvasir/ and dataset available from the Polyp-Grand Challenge: https://polyp.grand-challenge.org/
We follow the UNET code from this repository: (https://github.com/kkweon/UNet-in-Tensorflow). 

## Dataset
The dataset can be downloaded from the links. The directory structure should be as follows. 
```
  Dataset
    -> Images
    -> Ground Truth
```

## Training
To train UNET on dataset, type `python3 train.py`. 
