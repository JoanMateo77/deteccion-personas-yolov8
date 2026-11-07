# Detección y analítica de personas con YOLOv8

Detección de personas en tiempo real desde la webcam con YOLOv8 y OpenCV. Incluye seguimiento con IDs persistentes, conteo por línea virtual, alertas y un análisis posterior de la sesión con mapa de calor y tiempo de permanencia por persona.

Proyecto del curso Bases de Datos II de la Universidad Autónoma de Occidente (Cali, Colombia).

![Sistema profesional en ejecución](docs/img/06_sistema_profesional.png)

## Qué hace

**Cuaderno básico** (`deteccion_personas_basico.ipynb`)
- Detecta la clase `person` (COCO) en cada frame de la webcam y dibuja las cajas con su confianza.
- Mide FPS, tiempo de inferencia, personas por frame y confianza promedio.
- Exporta las métricas por frame a CSV y grafica su evolución.

**Cuaderno profesional** (`deteccion_personas_profesional.ipynb`)
- Seguimiento con `model.track(persist=True)`: cada persona conserva su ID entre frames y se dibuja su trayectoria.
- Conteo de entradas y salidas cuando el centro de una persona cruza una línea horizontal virtual.
- Alertas en pantalla con enfriamiento de 2 s: densidad alta (3 o más personas) o detección confiable (confianza promedio de 0,75 o más).
- Graba el video anotado en MP4 y exporta métricas por frame, incluidos los conteos y las alertas.

**Análisis posterior** (`scripts/procesar_mejora_real.py`)
- Vuelve a procesar el MP4 grabado con tracking, tomando 1 de cada 3 frames.
- Genera un mapa de calor de zonas transitadas: grilla de 64×36 con suavizado gaussiano (σ = 1,5).
- Estima el tiempo de permanencia de cada ID y grafica los 10 mayores.

## Resultados de las sesiones registradas

Datos tomados de los CSV y del JSON en `resultados/`. Cada sesión se grabó con una escena y una resolución distintas (960×540 y 1280×720), así que las cifras no son comparables entre sí. La ejecución fue en un Mac con Apple Silicon.

| Métrica | Básico | Profesional |
|---|---|---|
| Frames procesados | 5 162 | 2 416 |
| Detecciones totales | 15 126 | 4 248 |
| FPS promedio | 19,28 | 20,40 |
| Inferencia promedio | 26,19 ms | 31,73 ms |
| Confianza promedio | 0,77 | 0,83 |
| Entradas / salidas por línea | — | 9 / 9 |
| IDs únicos (análisis posterior) | — | 46 |
| Permanencia máxima estimada | — | 82,5 s (ID 48) |

### Capturas

| Detección básica | Frame del video anotado |
|---|---|
| ![Detección básica](docs/img/01_deteccion_basica_3personas.png) | ![Frame del MP4](docs/img/07_frame_del_mp4.png) |

| FPS por frame | Latencia de inferencia | Personas por frame |
|---|---|---|
| ![FPS](docs/img/03_fps_por_frame.png) | ![Latencia](docs/img/04_latencia_inferencia.png) | ![Personas](docs/img/05_personas_por_frame.png) |

| Mapa de calor de zonas | Top 10 de permanencia |
|---|---|
| ![Mapa de calor](docs/img/08_heatmap_zonas.png) | ![Permanencia](docs/img/09_dwell_time_top10.png) |

## Stack

- Python 3.10, JupyterLab
- Ultralytics YOLOv8n (pesos preentrenados en COCO)
- OpenCV para captura, dibujo y grabación de video
- pandas, NumPy, Matplotlib y SciPy para métricas y gráficas

## Estructura

```
.
├── deteccion_personas_basico.ipynb        # Detección y métricas
├── deteccion_personas_profesional.ipynb   # Tracking, conteo, alertas y grabación
├── scripts/
│   └── procesar_mejora_real.py            # Mapa de calor y permanencia sobre el MP4
├── resultados/                            # CSV y resumen JSON de las sesiones registradas
└── docs/img/                              # Capturas usadas en este README
```

## Cómo correrlo

```bash
conda create -n yolo python=3.10 -y
conda activate yolo
pip install ultralytics opencv-python pandas numpy matplotlib scipy jupyterlab
jupyter lab
```

1. En macOS, autoriza el acceso a la cámara para la terminal que lanza Jupyter (Configuración del Sistema → Privacidad y seguridad → Cámara).
2. Abre un cuaderno y ejecuta las celdas en orden. En la ventana de OpenCV, la tecla `q` termina la captura. Los pesos `yolov8n.pt` se descargan solos la primera vez.
3. El cuaderno profesional deja en la raíz `salida_deteccion_personas_profesional.mp4` y `metricas_deteccion_personas_profesional.csv`. Con esos archivos ya generados, corre el análisis posterior:

```bash
python scripts/procesar_mejora_real.py
```

Las imágenes y el `resumen_pro.json` se guardan en `resultados/`. El video no está en el repositorio porque supera el límite de 100 MB de GitHub.

## Decisiones técnicas

- **Filtrado de clase dentro del modelo.** El cuaderno profesional pasa `classes=[0]` a `model.track()`, así el modelo solo devuelve personas. El básico, en cambio, filtra después de la inferencia.
- **Conteo por cruce de línea con IDs.** Se guarda de qué lado de la línea estaba el centro de cada ID. El conteo sube solo cuando cambia de lado, y cada ID cuenta una sola vez por sentido. Así una persona que se mueve cerca de la línea no infla el conteo.
- **Alertas con enfriamiento.** Una alerta no se repite antes de 2 s, para no saturar la pantalla en cada frame.
- **Análisis fuera de línea.** El mapa de calor y la permanencia se calculan después, sobre el video grabado. El bucle en vivo se mantiene liviano.

## Limitaciones

- El seguimiento usa el tracker por defecto de `model.track()` de Ultralytics; no se ajustaron sus parámetros.
- La permanencia es una estimación: frames con aparición × 3 (por el submuestreo) / FPS promedio de la sesión.
- Los IDs pueden cambiar si una persona sale de cuadro o queda oculta, lo que reparte su permanencia entre varios IDs.
- No hay `requirements.txt` con versiones fijas ni pruebas automatizadas.

---

Hecho por [Joan Mateo Cardona](https://github.com/JoanMateo77).
