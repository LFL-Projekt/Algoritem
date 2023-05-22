import cv2
import numpy as np
import glob
from sklearn.model_selection import train_test_split
import os
import tensorflow as tf

def lbp(slika):
    # definirajmo sosede za vsako točko v matriki
    sosedje_indeksi = np.array([
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ])

    # pridobimo dimenzije slike
    visina, sirina = slika.shape

    # obrobimo sliko z vrednostmi roba
    obrobna_slika = np.pad(slika, pad_width=1, mode='edge')

    # pripravimo matriko za shranjevanje vrednosti sosedov
    vrednosti_sosedov = np.zeros((visina, sirina, 8), dtype=slika.dtype)

    # izračunamo vrednosti sosedov za vsako točko slike
    for i, (dy, dx) in enumerate(sosedje_indeksi):
        vrednosti_sosedov[..., i] = obrobna_slika[1 + dy : visina + 1 + dy, 1 + dx : sirina + 1 + dx]

    # pripravimo matriko za shranjevanje binarnih vzorcev
    sredina = slika
    binarni_vzorec = (vrednosti_sosedov >= sredina[..., np.newaxis]).astype(int)

    # pripravimo matriko za shranjevanje LBP vrednosti
    potence_dveh = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.uint8)
    lbp_vrednosti = np.sum(binarni_vzorec * potence_dveh, axis=-1)
    lbp_slika = np.zeros_like(slika, dtype=np.uint8)
    lbp_slika = lbp_vrednosti

    return lbp_slika