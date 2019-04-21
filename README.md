Naštudovať problematiku detekcie a sledovania ľudských tvárí. Analyzovať existujúce riešenia publikované v dostupnej odbornej literatúre. Vytvoriť databázu videozáznamov a hľadaných tvárí pre testovacie účely. Navrhnúť a implementovať metódu, ktorá bude schopná vyhľadať osobu vo viacerých videozáznamoch podľa zadaného vizuálneho vzoru t.j. tváre človeka. Vyhodnotiť dosiahnuté výsledky.

Popis súborov:
- centroid_tracker.py: tracker umožňujúci uchovanie ID pre osoby medzi jednotlivými frammami videa
- config.py: confirguračné premenné
- detect.py: funkcie pre detekciu postavy a tváre
- download_dataset.py: stiahne všetky nutné datasety a uloží ich do správneho formátu. Nemusí fungovať 100%, treba overiť, či sú súbory na správnom mieste (ako v configu)
- feature_extractors.py: funkcie na extrakciu príznakov obrázku
- feature_test.py: testovanie extrahovaných príznakov
- help_functions.py: pomocné funkcie, ako je načítanie obrázkov a podobne
- improve.py: umožnuje vylepšovať model na základe novo-získaných dát
- person_track.py: trieda obsahujúca informácie, na základne ktorých vieme osobu identifikovať
- play.py: prehrávanie videa a detekcia/respoznávanie osôb v ňom
- players.py: rôzne formáty video (obrázky, video, stream z kamery, video na youtube)
- recognize.py: funkcie umožňujuce rozpoznanie osôb na videu
- requirements.txt: nutné python knižnice
- retinex.py: filter na úpravu obrázky (nie môj kód)
- siamese_network.py: siamska konvolučná neurónová sieť
- test.py: testovanie natrénovaného modelu siamskej siete
- train.py: trénovanie modelu siamskej siete
- train_help_functions.py: pomocné funkcie na trénovanie (generovanie dát)

HOW TO RUN:  
install python3  
make new python environment  
install requirements  
(optional) install cuda, cuDNN and uncomment requirements if you want to use GPU
