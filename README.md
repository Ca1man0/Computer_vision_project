# Rilevamento di Mob in Minecraft - HOG + SVM vs YOLOv5-nano

Rilevamento e classificazione di quattro mob ostili di *Minecraft* (**Creeper, Skeleton, Spider, Zombie**) negli screenshot di gioco, con un confronto rigoroso tra due paradigmi della computer vision:

- **Pipeline A - Classica:** feature hand-crafted **HOG** + classificatore **SVM**, con detection a finestra scorrevole + piramide di immagini + NMS.
- **Pipeline B - Deep Learning:** **YOLOv5-nano** fine-tunato da COCO tramite transfer learning.

**Autore:** Aiman Hamdouni

---

## 1. Panoramica

Il progetto affronta un problema di **object detection multi-classe** controllato ma realistico: ogni mob deve essere sia *localizzato* (bounding box) sia *classificato*. Il dataset contiene volutamente difficoltà reali (sbilanciamento delle classi, oggetti molto piccoli/lontani e forti variazioni di illuminazione e colore dovute ai biomi) il che lo rende un proxy fedele per compiti come videosorveglianza o monitoraggio della fauna, senza raccogliere immagini di persone reali.

Entrambe le pipeline sono addestrate e valutate sugli **stessi dati con le stesse metriche**, così i risultati rispondono a una domanda ingegneristica concreta: *quando conviene un modello deep pesante e quando basta una pipeline classica leggera?*

Documento completo: **Documento_analisi_tecnica ver 2.0.docx** 

---

## 2. Dataset

- **Fonte:** Kaggle - [`dracotlw/minecraft-mobs-yolo-dataset`](https://www.kaggle.com/datasets/dracotlw/minecraft-mobs-yolo-dataset) (scaricato automaticamente tramite `kagglehub`).
- **Dimensione:** 2.585 screenshot (2.068 train / 517 validation), tutti 1920×1080, 8 bit.
- **Formato:** annotazioni YOLO (`class_id x_center y_center width height`, normalizzate).
- **Classi:** `0 Creeper`, `1 Skeleton`, `2 Spider`, `3 Zombie`.
- **Sbilanciamento:** 2,34× (Zombie 914 => Spider 390 istanze nel train).
- **Immagini background:** 488 (23,6%) senza mob - usate come hard negative.

Non serve scaricare i dati manualmente: il primo notebook li recupera con `kagglehub.dataset_download(...)`.

---

## 3. Struttura del repository

```
minecraft-mob-detection/
├── README.md
├── requirements.txt
├── docs/
│   └──Documento_Analisi_Tecnica.docx      # report tecnico completo (5 sezioni) - Word
│  
└── notebooks/
    ├── MOD_1_dataset_exploration.ipynb     # statistiche dataset, bilanciamento, dimensioni bbox, istogrammi
    ├── MOD_2_preprocessing.ipynb           # spazi colore, denoising, point op, equalizzazione, augmentation, crop, edge
    ├── MOD_3_HOG_SVM_pipeline.ipynb        # feature HOG + k-NN / SVM(Lineare/RBF) + detection sliding-window
    ├── MOD_4_yolo_training.ipynb           # fine-tuning di YOLOv5-nano da COCO
    └── MOD_5_evaluation.ipynb              # IoU/mAP manuali, curve PR, confronto HOG+SVM vs YOLO
```

Esegui i notebook **in ordine (MOD 1 => MOD 5)**; ogni modulo si basa sul precedente.

---

## 4. Pipeline e architettura

### Preprocessing (MOD 2)
Due pipeline specifiche progettate dopo uno studio comparativo:

| Target | Preprocessing |
|---|---|
| **HOG + SVM** | scala di grigi => equalizzazione istogramma su V (HSV) => resize crop a **64×128**. Niente blur (indebolirebbe i gradienti su cui si basa HOG). |
| **YOLOv5** | **bilaterale** leggero => RGB, normalizzato in [0,1] => **letterbox 640×640**. Augmentation (mosaic, HSV, flip) gestita dal framework. |

### Pipeline A - HOG + SVM (MOD 3)
1. Estrazione di 2.400 crop (1.200 positivi @ 300/classe + 1.200 negativi di background), 64×128 in scala di grigi.
2. Descrittore **HOG** => vettore di feature a **3.780-D** per crop (celle 8×8, 9 bin, norm. a blocchi 2×2).
3. Split stratificato 75/25 + `StandardScaler`.
4. Confronto **k-NN** (baseline), **SVM-Lineare**, **SVM-RBF**.
5. **Detection:** piramide di immagini => finestra scorrevole 64×128 => HOG+SVM => **NMS** (IoU).

### Pipeline B - YOLOv5-nano (MOD 4)
Detector single-shot fine-tunato dai pesi COCO (transfer learning):
`backbone CSPDarknet => neck PANet (fusione multi-scala) => detection head`.
Loss = **CIoU (box) + BCE (objectness) + BCE (classe)**, ottimizzatore SGD (momentum 0,937), **50 epoche (~32 min su Tesla T4)**. Modello finale: ~1,76M parametri, 4,1 GFLOPs, **3,8 MB**.

### Valutazione (MOD 5)
Protocollo condiviso sullo stesso validation set: **IoU** (TP se IoU ≥ 0,5, stessa classe), **Precision / Recall / F1**, **AP** per classe (interpolazione PASCAL-VOC), **mAP** e latenza di inferenza.

---

## 5. Sintesi dei risultati

**Selezione del classificatore (feature HOG, a livello di crop)**

| Modello | Accuracy |
|---|---|
| k-NN (k=1) | 0,843 |
| SVM - Lineare | 0,835 |
| **SVM - RBF** | **0,910** selezionato |

**Training di YOLOv5-nano (migliore @ epoca 42):** Precision **0,956** · Recall **0,840** · mAP@0,5 **0,912** · mAP@0,5:0,95 **0,675** - nessun overfitting (le loss train/val convergono).

**Confronto diretto (stesse 4 classi, protocollo IoU condiviso)**

| Metrica | HOG + SVM | YOLOv5-nano | Vincitore |
|---|---|---|---|
| Precision | 0,675 | **0,745** | YOLO |
| Recall | 0,600 | **0,634** | YOLO |
| F1-score | 0,634 | **0,685** | YOLO |
| mAP@0,5 | - | **0,787** | YOLO |
| Inferenza | 1,4 ms / crop | 138,3 ms / img | vedi nota |
| Ingombro | minimo, no GPU | 3,8 MB | HOG+SVM |

**In sintesi:** YOLOv5-nano vince su ogni metrica di qualità; l'unico vantaggio reale di HOG+SVM è l'ingombro minimo e il funzionamento senza GPU. Resta comunque una baseline utile e interpretabile.

---

## 6. Setup e istruzioni per l'esecuzione

Il progetto è stato sviluppato su Colab (Python 3.10, Tesla T4). Apri ogni notebook in Colab ed eseguilo dall'inizio alla fine:

<!-- Sostituisci <USER>/<REPO> con il tuo percorso GitHub dopo il push -->
- MOD 1 - Esplorazione dataset: `https://colab.research.google.com/github/<USER>/<REPO>/blob/main/notebooks/MOD_1_dataset_exploration.ipynb`
- MOD 2 - Preprocessing: `.../notebooks/MOD_2_preprocessing.ipynb`
- MOD 3 - HOG + SVM: `.../notebooks/MOD_3_HOG_SVM_pipeline.ipynb`
- MOD 4 - Training YOLOv5: `.../notebooks/MOD_4_yolo_training.ipynb`  *(imposta Runtime -> GPU)*
- MOD 5 - Valutazione: `.../notebooks/MOD_5_evaluation.ipynb`

> Colab include già la maggior parte delle dipendenze. Ogni notebook installa il resto (es. `kagglehub`) nella prima cella. Per **MOD 4** abilita una GPU: *Runtime -> Cambia tipo di runtime -> GPU T4*.

Per MOD 4 (fine-tuning) è **fortemente consigliata una GPU**. MOD 1–3 e MOD 5 girano comodamente su CPU.

### Accesso al dataset
I notebook scaricano il dataset tramite `kagglehub`. Se vengono richieste le credenziali, inserisci il tuo token API `kaggle.json` come descritto nella [documentazione API di Kaggle](https://www.kaggle.com/docs/api). Nessun download manuale richiesto.

---

## 7. Riproducibilità e note
- Seed casuali fissi e split stratificato 80/20 (immagini) / 75/25 (crop).
- **MOD 4** contiene un training già eseguito; il ri-addestramento richiede ~32 min su una T4 (una nota nel notebook segnala la cella lunga).
