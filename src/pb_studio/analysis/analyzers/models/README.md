# DNN Face Detector Models

Dieser Ordner enthält die Pre-Trained Models für den OpenCV DNN Face Detector.

## Benötigte Dateien

1. **deploy.prototxt** - Netzwerk-Architektur (Caffe Prototxt)
2. **res10_300x300_ssd_iter_140000.caffemodel** - Pre-Trained Weights

## Download

Lade die Modelle von GitHub herunter:

```bash
# deploy.prototxt
curl -o deploy.prototxt https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt

# Caffemodel (~10 MB)
curl -o res10_300x300_ssd_iter_140000.caffemodel https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel
```

## Fallback

Wenn die DNN-Modelle nicht verfügbar sind, verwendet PB Studio automatisch den Haar Cascade Classifier als Fallback (bereits in OpenCV enthalten).

**DNN Vorteile:**
- Höhere Accuracy (~95% vs ~70%)
- Weniger False Positives
- Robuster bei Skalierung/Rotation
- Schneller bei GPU-Beschleunigung

**Haar Cascade Nachteile:**
- Veraltet (2001)
- Viele False Positives
- Schlechter bei Profile-Ansichten
- Langsamer bei CPU-Only
