# Predicción de precios de viviendas en Ames, Iowa

Evaluación Parcial 3, MLY1101 Machine Learning (DuocUC). Caso B: predicción de precios de viviendas.

Integrantes: Nicolas Osses y Javier Sagredo.

El proyecto estima el precio de venta de una vivienda en Ames a partir de sus características físicas, de calidad y de ubicación, como apoyo al perito tasador. Esta entrega toma los cuatro modelos de la EP2 (Ridge, Lasso, Random Forest y XGBoost), ajusta sus hiperparámetros con validación cruzada, prueba ensambles (bagging, boosting, votación y stacking), mide la capacidad de generalización y elige el modelo final con una regla fijada antes de abrir el test.

El informe técnico completo está en [`docs/informe_tecnico.md`](docs/informe_tecnico.md).

## Resultado

El modelo final es Lasso con `alpha` = 0,000174, que usa 105 de 138 columnas. En el test (582 casas) cumple los seis indicadores del criterio de salida de la EP1:

| KPI | Umbral | Test real | Venta normal impuesta |
|---|---|---|---|
| R² | ≥ 0,85 | 0,948 | 0,949 |
| RMSE | ≤ 30.000 USD | 19.241 | 19.084 |
| MAE | ≤ 20.000 USD | 12.407 | 12.462 |
| MAPE | ≤ 12 % | 6,77 % | 6,93 % |
| Brecha de R² train-test | ≤ 5 puntos | −0,9 | −1,0 |
| Brecha de MAPE entre barrios | ≤ 8 puntos | 6,1 | 6,2 |

Lo que cambió respecto de la EP2:

- Con la búsqueda de hiperparámetros, XGBoost dejó de sobreajustar (brecha de R² de 8,2 a 3,9 puntos en validación) y bajó su RMSE de validación de 22.351 a 19.809 USD. Random Forest pasó a cumplir la brecha, pero sin mejorar su error.
- El stacking de Lasso con XGBoost tuvo el menor error del proyecto: 19.203 USD en validación y 17.903 en test. Le gana a Lasso en todos los folds, pero por menos que la variación entre folds y por 0,66 puntos de R², bajo el punto que exige la regla para preferir un modelo menos interpretable.
- Se mantuvo Lasso porque cada estimación se puede descomponer en el aporte de cada variable y porque su resultado no depende del sistema operativo ni de cómo se buscaron los hiperparámetros.

La función `tasar(casa)` del notebook (sección 51) entrega el valor estimado, un rango orientativo, el desglose por variable y alertas para derivar a revisión manual.

## Contenido

```
├── data/
│   ├── raw/Ames_Iowa_Housing_Dataset.csv     datos originales
│   └── processed/                            ames_clean.csv (salida de la EP1), conjuntos por modelo y segmentos.csv
├── docs/
│   └── informe_tecnico.md                    informe técnico
├── images/                                   figuras de la parte 3 (p3_*) y de la parte 4 (p4_*)
├── models/
│   ├── busquedas/                            resultados de cada búsqueda de hiperparámetros (36 JSON)
│   ├── parte4/                               ensambles, folds nuevos, validación anidada, curvas y resultados del test
│   ├── mejores_params.json                   hiperparámetros elegidos para los cuatro modelos
│   ├── modelo_final.joblib                   modelo final entrenado con todo train
│   └── modelo_final.json                     ficha del modelo final (parámetros, métricas, forma de uso)
├── notebooks/
│   └── Housing_Prices_ML_P3_v5.ipynb         notebook completo
├── requirements.txt
└── README.md
```

El notebook tiene cuatro partes:

| Parte | Secciones | Qué hace |
|---|---|---|
| 1 | 1 a 18 | Prepara los datos para cada familia de modelos y escribe los CSV de `data/processed/` |
| 2 | 19 a 27 | Compara los cuatro modelos con parámetros iniciales y elige Lasso (EP2) |
| 3 | 28 a 40 | Busca los hiperparámetros de los cuatro modelos y guarda la evidencia en `models/` |
| 4 | 41 a 52 | Ensambles, generalización, test final, selección, función de tasación y modelo final |

## Cómo ejecutar

### En local

Hace falta Python 3.12 o 3.13: numpy 2.1.3 no tiene paquetes para Python 3.14.

Windows (PowerShell):

```powershell
py -3.13 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m ipykernel install --user --name housing-p3 --display-name "Housing P3 (.venv)"
$env:PYTHONUTF8 = "1"
jupyter notebook
```

Linux o macOS:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name housing-p3 --display-name "Housing P3 (.venv)"
jupyter notebook
```

Después, abrir `notebooks/Housing_Prices_ML_P3_v5.ipynb` con el kernel «Housing P3 (.venv)» y ejecutar todas las celdas. En VS Code basta con elegir ese kernel. La celda «Entorno, rutas y datos» comprueba que numpy, pandas, scikit-learn y xgboost tengan las versiones de `requirements.txt`, y se detiene si no coinciden.

En Windows, `PYTHONUTF8=1` hace que Python escriba los JSON en UTF-8. Sin esa variable, `models/mejores_params.json` queda en cp1252 (ver la sección 15.5 del informe).

### En Google Colab

1. Abrir el notebook en Colab y ejecutar la primera celda. Pide subir `requirements.txt` y, si las versiones de Colab no son las exactas, las instala y reinicia la sesión. Después se vuelve a ejecutar todo desde el inicio.
2. La celda de entorno pide subir `data/processed/ames_clean.csv` si no lo encuentra.
3. Colab no tiene las carpetas `models/busquedas/` ni `models/parte4/`, así que el notebook recalcula todas las búsquedas y la parte 4. Con los 2 núcleos de Colab toma bastante más que en local; no lo medimos.
4. La última celda comprime `models/` e `images/` para descargarlos antes de que termine la sesión.

### Resultados guardados y tiempos

Las búsquedas de la parte 3 y los cálculos largos de la parte 4 se guardan en JSON dentro de `models/` y el notebook los carga si existen. Con esos archivos, la ejecución completa tomó 6 minutos en Linux con 16 núcleos. Para recalcular todo, poner `REPETIR_BUSQUEDAS = True` (sección 28) y `REPETIR_PARTE4 = True` (sección 41). Solo las búsquedas de la parte 3 tomaron 18 minutos con 12 núcleos.

El test se abre una sola vez, en la sección 48. Esa celda deja una bandera que impide ejecutarla de nuevo sin reiniciar el kernel.

## Diferencias entre sistemas operativos

Las cifras del notebook y del informe salen de una ejecución completa en Windows del 20-11-2026. Con las mismas versiones de las librerías, Linux (y por lo tanto Colab) da otros resultados en XGBoost cuando submuestrea filas o columnas:

- Ridge y Lasso dan exactamente lo mismo, así que el modelo final y sus métricas no cambian.
- Random Forest cambia en pocos dólares.
- XGBoost, la votación y el stacking cambian en cientos de dólares. La búsqueda de la parte 3 eligió en Linux otra configuración de XGBoost (profundidad 3 y 1.500 árboles, con 20.011 USD, frente a profundidad 2 y 2.000 árboles en Windows).
- En Linux, la votación y el stacking dejan de cumplir la equidad entre barrios con venta normal (8,5 y 8,6 puntos frente a un máximo de 8; en Windows, 7,4 y 7,5).

El modelo elegido es el mismo en los dos sistemas. La sección 15.3 del informe tiene la comparación completa y qué hicimos con ella.

## Datos

Ames, Iowa Housing Dataset (De Cock, 2011): 2.930 ventas de viviendas entre 2006 y 2010, con 82 columnas. La EP1 dejó `data/processed/ames_clean.csv` (2.925 casas, 80 columnas, sin nulos), que es el punto de partida del notebook.
