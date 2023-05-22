## Opis
Ta Python program implementira algoritma LBP (Local Binary Patterns) in HOG (Histogram of Oriented Gradients) ter uporablja TensorFlow za učenje. Algoritma 
LBP in HOG sta pogosto uporabljena za zaznavanje in prepoznavanje objektov v slikah.

Algoritem LBP je preprosta metoda za opisovanje teksture v sliki. Temelji na primerjavi intenzitet sosednjih pikslov in določanju binarne vrednosti za 
vsak piksel, kar ustvari lokalni binarni vzorec (LBP). Ta vzorec lahko nato uporabimo za opisovanje teksture in razpoznavanje objektov.

HOG (Histogram of Oriented Gradients) je metoda za opisovanje objektov v sliki na podlagi njihovih lokalnih gradientov. 
Metoda deluje tako, da izračuna magnitudo in smer gradientov v majhnih oknih po sliki. Nato se te vrednosti združijo v histograme orientiranih gradientov, 
ki vsebujejo informacije o prisotnosti določenih vzorcev v sliki. HOG je učinkovit pri zaznavanju objektov, ki se razlikujejo po teksturi in obliki.

Ta program uporablja knjižnico TensorFlow, ki je odprtokodna knjižnica za strojno učenje. TensorFlow omogoča izgradnjo in usposabljanje nevronskih mrež 
ter implementacijo različnih algoritmov za obdelavo slik. S pomočjo TensorFlowa lahko algoritme LBP in HOG uporabimo za učenje modelov, ki lahko nato prepoznavajo obraze v slikah.
