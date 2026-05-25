# Learning path del progetto ADAS

## Fase 1 — Percezione oggetti
Obiettivo: caricare YOLOv8, leggere webcam/video, filtrare classi rilevanti: car, truck, bus, motorcycle, bicycle, person.

Da capire bene:
- cos'è un bounding box;
- confidence threshold;
- differenza tra detection e tracking;
- limiti del dataset COCO per guida autonoma.

## Fase 2 — Rilevamento corsie
Obiettivo: trovare linee della carreggiata con pipeline classica OpenCV.

Da capire bene:
- ROI, cioè regione di interesse;
- Canny edge detection;
- Hough transform;
- perché servono filtri geometrici sulle linee.

## Fase 3 — Rischio collisione
Obiettivo: stimare se un oggetto nella nostra corsia si sta avvicinando.

Concetto chiave: Time-To-Collision visuale.
Se il bounding box cresce rapidamente, probabilmente la distanza si sta riducendo.
Questa è solo una stima: un sistema reale usa anche radar, LiDAR, stereo camera o sensori veicolo.

## Fase 4 — Decisione ADAS
Obiettivo: trasformare percezione e rischio in stati chiari:
- nessun rischio;
- warning;
- emergency stop simulato;
- lane departure warning.

## Fase 5 — Verso un prodotto vero
Per avvicinarsi a un sistema vendibile servono:
- dataset stradali reali e annotati;
- validazione quantitativa;
- logging e replay;
- test unitari, test scenario-based, simulazione;
- architettura ROS 2;
- safety case, hazard analysis e conformità automotive.
