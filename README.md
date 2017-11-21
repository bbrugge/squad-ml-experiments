# SQuAD ML experiments

experiments on the Stanford Question Answering Dataset (https://rajpurkar.github.io/SQuAD-explorer/)

based on:
https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html

## install

```
pip install tensorflow-gpu keras numpy
```

## usage

```
$ python squad_train.py
$ python squad_guess.py
> Who won superbowl 50?
...
top answers:
 0.31140444    Denver Broncos
 0.04180731    Coldplay
 0.01792004    New England Patriots
 0.00632825    Lady Gaga
 0.00570256    Von Miller
 0.00536138    Coldplay.
 0.00530134    Tiffany & Co
 0.00515696    Jim Gray
 0.00478155    Jonathan Stewart
 0.00464156    Kublai Khan
 0.00391972    six
 0.00361646    third
 0.00303506    Ed Lee
 0.00294518    Miller
 0.00293098    Temüjin and his brother Khasar
 0.00275560    Stewart
 0.00254358    the New England Patriots
 0.00248989    Labour
 0.00227957    Kevin Harlan
 0.00226449    Osama bin Laden
```
