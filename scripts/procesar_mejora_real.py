"""
Re-procesa salida_deteccion_personas_profesional.mp4 con YOLOv8 + tracking
para obtener centroides reales por ID y generar, en resultados/:
  - heatmap_zonas.png
  - dwell_time_top10.png
  - frame_representativo.png (frame extraído del video)

Adicionalmente analiza el CSV profesional para producir:
  - resumen_pro.json

Entradas: el video y el CSV que genera el notebook profesional en la raíz
del repositorio. Si el CSV no está en la raíz, se usa el de resultados/.
"""
import json
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from scipy.ndimage import gaussian_filter

RAIZ = Path(__file__).resolve().parent.parent
CARPETA = RAIZ / "resultados"
CARPETA.mkdir(exist_ok=True)
VIDEO = RAIZ / "salida_deteccion_personas_profesional.mp4"
CSV_PRO = RAIZ / "metricas_deteccion_personas_profesional.csv"
if not CSV_PRO.exists():
    CSV_PRO = CARPETA / "metricas_deteccion_personas_profesional.csv"

# ============== 1) FRAME REPRESENTATIVO DEL MP4 ==============
print(">> Extrayendo frame representativo del MP4...")
cap = cv2.VideoCapture(str(VIDEO))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"   Total frames del video: {total}")

# Buscamos un frame en el segundo cuarto del video (suele haber acción)
target = total // 3
cap.set(cv2.CAP_PROP_POS_FRAMES, target)
ret, frame = cap.read()
if ret:
    cv2.imwrite(str(CARPETA / "frame_representativo.png"), frame)
    print(f"   Frame {target} guardado como frame_representativo.png")
cap.release()

# ============== 2) RE-PROCESO CON YOLO TRACKING ==============
print(">> Cargando YOLOv8n...")
modelo = YOLO("yolov8n.pt")

print(">> Re-procesando MP4 con tracking (esto toma ~1-2 min)...")
cap = cv2.VideoCapture(str(VIDEO))
FRAME_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
FRAME_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"   Resolución: {FRAME_W}x{FRAME_H}")

centros = []  # lista de (cx, cy) acumulados
permanencia = defaultdict(int)  # id -> count de frames donde apareció
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    # Sub-muestreo: procesamos 1 de cada 3 frames para acelerar
    if frame_idx % 3 != 0:
        continue

    resultados = modelo.track(
        source=frame,
        persist=True,
        conf=0.40,
        iou=0.45,
        classes=[0],
        verbose=False
    )

    if len(resultados) > 0 and resultados[0].boxes is not None:
        boxes = resultados[0].boxes
        if boxes.id is not None:
            ids = boxes.id.cpu().numpy().astype(int)
            xyxy = boxes.xyxy.cpu().numpy()
            for tid, (x1, y1, x2, y2) in zip(ids, xyxy):
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                centros.append((cx, cy))
                permanencia[int(tid)] += 1

    if frame_idx % 300 == 0:
        print(f"   Progreso: {frame_idx}/{total} ({100*frame_idx/total:.0f}%)")

cap.release()
print(f">> Centroides totales: {len(centros)}")
print(f">> IDs únicos: {len(permanencia)}")

# ============== 3) HEATMAP ==============
print(">> Generando heatmap...")
GRID_W, GRID_H = 64, 36
heatmap = np.zeros((GRID_H, GRID_W), dtype=np.float32)

for cx, cy in centros:
    gx = int(min(max(cx / FRAME_W * GRID_W, 0), GRID_W - 1))
    gy = int(min(max(cy / FRAME_H * GRID_H, 0), GRID_H - 1))
    heatmap[gy, gx] += 1

heatmap_blur = gaussian_filter(heatmap, sigma=1.5)

plt.figure(figsize=(12, 6))
plt.imshow(heatmap_blur, cmap='inferno', aspect='auto',
           extent=[0, FRAME_W, FRAME_H, 0])
plt.title('Mapa de calor de zonas más transitadas\n(Sub-muestreo cada 3 frames)',
          fontsize=12)
plt.colorbar(label='Detecciones acumuladas (densidad)')
plt.xlabel('Eje X (píxeles)')
plt.ylabel('Eje Y (píxeles)')
plt.tight_layout()
plt.savefig(CARPETA / "heatmap_zonas.png", dpi=150, bbox_inches='tight')
plt.close()

# ============== 4) DWELL-TIME ==============
print(">> Generando dwell-time...")
df_csv = pd.read_csv(CSV_PRO)
fps_prom = df_csv['fps'].mean()

# Cada conteo en permanencia representa 3 frames (por el sub-muestreo)
dwell_seg = {tid: (cnt * 3) / fps_prom for tid, cnt in permanencia.items()}
top = sorted(dwell_seg.items(), key=lambda x: -x[1])[:10]

ids_str = [f"ID {t[0]}" for t in top]
secs = [t[1] for t in top]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(ids_str, secs, color='#A81223')
ax.set_title(f'Top 10 identificadores por dwell-time\n'
             f'(Total de IDs únicos en la sesión: {len(permanencia)})',
             fontsize=12)
ax.set_xlabel('Segundos en escena')
ax.set_ylabel('ID de tracking')
ax.invert_yaxis()
for bar, val in zip(bars, secs):
    ax.text(val + 0.2, bar.get_y() + bar.get_height() / 2,
            f'{val:.1f}s', va='center', fontsize=9)
plt.tight_layout()
plt.savefig(CARPETA / "dwell_time_top10.png", dpi=150, bbox_inches='tight')
plt.close()

# ============== 5) RESUMEN PRO PARA TABLA WORD ==============
resumen = {
    "frames_procesados": int(len(df_csv)),
    "detecciones_totales": int(df_csv['cantidad_personas'].sum()),
    "fps_promedio": float(df_csv['fps'].mean()),
    "tiempo_inferencia_promedio_ms": float(df_csv['tiempo_inferencia_ms'].mean()),
    "personas_promedio_por_frame": float(df_csv['cantidad_personas'].mean()),
    "confianza_promedio": float(df_csv['confianza_promedio_frame'].dropna().mean()),
    "entrada_total": int(df_csv['conteo_entrada'].max()),
    "salida_total": int(df_csv['conteo_salida'].max()),
    "ids_unicos_tracking": int(len(permanencia)),
    "id_mayor_permanencia": int(top[0][0]) if top else 0,
    "dwell_max_segundos": float(top[0][1]) if top else 0.0,
}

with open(CARPETA / "resumen_pro.json", "w") as f:
    json.dump(resumen, f, indent=2)

print("\n>> RESUMEN PROFESIONAL")
for k, v in resumen.items():
    if isinstance(v, float):
        print(f"   {k}: {v:.4f}")
    else:
        print(f"   {k}: {v}")

print("\n✓ Procesamiento completo.")
print("   Generados:")
for f in ["frame_representativo.png", "heatmap_zonas.png",
          "dwell_time_top10.png", "resumen_pro.json"]:
    p = CARPETA / f
    if p.exists():
        print(f"     - {f} ({p.stat().st_size/1024:.1f} KB)")
