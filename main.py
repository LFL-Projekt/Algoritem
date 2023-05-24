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

def hog(sivinska_slika, velikost_celice, velikost_bloka, stevilo_predalov):
    visina, sirina = sivinska_slika.shape

    gx = np.zeros_like(sivinska_slika, dtype=np.float32)
    gy = np.zeros_like(sivinska_slika, dtype=np.float32)

    gx[:, :-1] = np.diff(sivinska_slika, n=1, axis=1)
    gy[:-1, :] = np.diff(sivinska_slika, n=1, axis=0)

    gradient_magnitude = np.sqrt(gx ** 2 + gy ** 2)
    gradient_orientacija = np.arctan2(gy, gx) * (180.0 / np.pi) % 180.0

    stevilo_celic_x = sirina // velikost_celice
    stevilo_celic_y = visina // velikost_celice

    histogram = np.zeros((stevilo_celic_y, stevilo_celic_x, stevilo_predalov))

    for y in range(stevilo_celic_y):
        for x in range(stevilo_celic_x):
            y_zacetek = y * velikost_celice
            y_konec = (y + 1) * velikost_celice
            x_zacetek = x * velikost_celice
            x_konec = (x + 1) * velikost_celice

            celica_magnitude = gradient_magnitude[y_zacetek:y_konec, x_zacetek:x_konec]
            celica_orientacija = gradient_orientacija[y_zacetek:y_konec, x_zacetek:x_konec]

            hist, _ = np.histogram(celica_orientacija, bins=stevilo_predalov, range=(0, 180),
                                   weights=celica_magnitude)

            histogram[y, x, :] = hist / np.sqrt(np.sum(hist ** 2) + 1e-6)

    korak_bloka = velikost_celice // velikost_bloka
    stevilo_blokov_x = (stevilo_celic_x - velikost_bloka) // korak_bloka + 1
    stevilo_blokov_y = (stevilo_celic_y - velikost_bloka) // korak_bloka + 1

    hog_opis = np.zeros((stevilo_blokov_y, stevilo_blokov_x, velikost_bloka, velikost_bloka, stevilo_predalov))

    for y in range(stevilo_blokov_y):
        for x in range(stevilo_blokov_x):
            y_zacetek = y * korak_bloka
            y_konec = y_zacetek + velikost_bloka
            x_zacetek = x * korak_bloka
            x_konec = x_zacetek + velikost_bloka

            blok_histogram = histogram[y_zacetek:y_konec, x_zacetek:x_konec, :]
            hog_opis[y, x, :, :, :] = blok_histogram / np.sqrt(np.sum(blok_histogram ** 2) + 1e-6)

    return hog_opis.flatten()

# Enkodirajte oznake z eno-hot kodiranjem
y_train_encoded = tf.keras.utils.to_categorical(y_train, 2)
y_test_encoded = tf.keras.utils.to_categorical(y_test, 2)

# Definirajte arhitekturo nevronske mreže
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(2, activation='softmax')
])

# Skompilirajte model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Učite model
model.fit(X_train, y_train_encoded, epochs=10, batch_size=32, validation_data=(X_test, y_test_encoded))

# Ocena modela
test_loss, test_acc = model.evaluate(X_test, y_test_encoded)
print('Test accuracy:', test_acc)

# Napovedovanje značilk
predictions = model.predict(vect)