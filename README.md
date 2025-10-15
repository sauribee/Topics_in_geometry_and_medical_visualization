# Topics in Geometry and Medical Visualization

Implementaciones en **Python** de métodos geométricos (interpolación 1D y curvas paramétricas 2D con splines) y un **pipeline de visualización médica** (segmentación de contornos óseos y ajuste con Bézier/B‑splines) para el curso *Tópicos en Geometría y Visualización Médica*.

## Tabla de contenido
- [Estructura del repositorio](#estructura-del-repositorio)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Ejecución rápida (quickstart)](#ejecución-rápida-quickstart)
- [Modo *headless* (sin GUI)](#modo-headless-sin-gui)
- [Visualización médica (plan de integración)](#visualización-médica-plan-de-integración)
- [Pruebas y reproducibilidad](#pruebas-y-reproducibilidad)
- [Licencia](#licencia)

## Estructura del repositorio
```
Topics_in_geometry_and_medical_visualization/
├── assignment_1/        # Interpolación con splines en 1D (lineal, cuadrática, etc.)
├── assignment_2/        # Curvas paramétricas 2D con splines (p. ej. elipse)
├── assignment_3/        # (a completar) Bézier y B‑splines: De Casteljau / De Boor, C^k
├── assignment_4/        # (a completar) Pipeline end‑to‑end de visualización médica
├── requirements.txt     # Dependencias del proyecto
├── INSTALL.md           # Notas de instalación (opcional)
├── setup.sh             # Script de conveniencia (opcional)
└── README.md            # Este archivo
```
Cada *assignment* puede incluir su propio `README.md` con detalles adicionales.

## Requisitos
Se recomienda **Python 3.12** (o 3.11). Evitar por ahora 3.13 por compatibilidades de paquetes de geometría que aún no publican *wheels* estables.
Dependencias base (se fijarán en `requirements.txt` en el siguiente paso de la checklist):
- `numpy`, `scipy`, `matplotlib`
- (Próximo) `scikit-image`, `opencv-python` (segmentación y morfología)
- (Próximo) `pydicom` (carga de DICOM), `pillow` (E/S imágenes)

## Instalación
```bash
# 1) Clonar el repositorio
git clone https://github.com/sauribee/Topics_in_geometry_and_medical_visualization.git
cd Topics_in_geometry_and_medical_visualization

# 2) Crear y activar entorno virtual (venv)
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# 3) Instalar dependencias
pip install -r requirements.txt
```

> Alternativa con Conda (opcional): crear `environment.yml` y usar `conda env create -f environment.yml` (lo añadiremos en la checklist).

## Ejecución rápida (quickstart)

### Assignment 1 (1D)
Ejecuta el script/módulo del *assignment* con parámetros por defecto y guarda salidas en `assignment_1/output/` (si aplica). Si el *assignment* define `main.py`, típicamente:
```bash
python -m assignment_1.main --save_only
```
> Revisa `assignment_1/README.md` para opciones disponibles (nodos, tipo de spline, etc.).

### Assignment 2 (2D)
Ejemplo típico de elipse paramétrica (guardando las figuras sin abrir ventanas):
```bash
python -m assignment_2.main --example ellipse --save_only
```
Las imágenes se guardarán en `assignment_2/output/` (o la carpeta definida por el *assignment*).

## Modo *headless* (sin GUI)
Para servidores sin interfaz gráfica:
- Usa `--save_only` en los comandos anteriores (cuando esté implementado).
- O fuerza el *backend* no interactivo de Matplotlib:
  ```bash
  export MPLBACKEND=Agg        # Linux/macOS
  setx MPLBACKEND Agg          # Windows PowerShell (persistente para futuras sesiones)
  ```
Los gráficos se guardarán a archivos (`.png`, `.pdf`) en la carpeta de salida correspondiente.

## Visualización médica (plan de integración)

Se añadirá un nuevo paquete `medvis/` con el pipeline completo **imagen → segmento → contorno → spline → muestreo equidistante**:

```
medvis/
├── io.py         # Lectura de PNG/JPG y (opcional) DICOM con pydicom
├── segment.py    # Segmentación básica: umbral/Canny + morfología + limpieza
├── curvefit.py   # Ajuste spline/Bézier al contorno; reparametrización por longitud de arco
├── sampling.py   # Muestreo equidistante sobre la curva ajustada
└── viz.py        # Utilidades de visualización y guardado en modo headless
```