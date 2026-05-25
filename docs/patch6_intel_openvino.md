# Patch 6 - Intel Iris Xe / OpenVINO

Il PC rileva `Intel(R) Iris(R) Xe Graphics`, quindi non c'è CUDA. CUDA è la strada NVIDIA; su Intel la pipeline corretta è OpenVINO.

## Comandi

```powershell
python -m pip install -e .
python scripts/check_accelerators.py
python scripts/export_openvino.py --model models/yolov8n.pt --half --imgsz 416
python -m adas.main --config configs/realtime_iris_xe.yaml --source "data/dashcam1.mp4"
```

## Note

- `--gpu` = NVIDIA CUDA, non Intel Iris Xe.
- `--intel-gpu` = OpenVINO su Intel GPU, richiede modello esportato OpenVINO.
- `--intel-cpu` = OpenVINO su Intel CPU, utile se OpenVINO non vede la GPU.
- Il modello `.pt` va esportato prima in `models/yolov8n_openvino_model`.
- Su OpenVINO il tracking viene disattivato di default per stabilità.

## Diagnostica

```powershell
python scripts/check_accelerators.py
```

Output desiderato:

```text
openvino devices: ['CPU', 'GPU']
```

Se compare solo `CPU`, aggiorna i driver Intel Graphics.
