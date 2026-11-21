# Predicción de precios de viviendas en Ames, Iowa

Informe técnico. Evaluación Parcial 3, MLY1101 Machine Learning (DuocUC). Caso B: predicción de precios de viviendas.

Integrantes: Nicolas Osses y Javier Sagredo.

Notebook: `notebooks/Housing_Prices_ML_P3_v5.ipynb`. Las secciones que cita este informe (por ejemplo, "sección 32") son las del notebook.

## Contenido

1. Problema de negocio
2. Objetivos
3. Fuentes de datos
4. Análisis exploratorio
5. Preparación y transformación de los datos
6. Selección de modelos
7. Variables finales por modelo
8. Implementación
9. Modelos con parámetros iniciales (EP2)
10. Optimización de hiperparámetros y validación cruzada
11. Técnicas de ensamble
12. Capacidad de generalización
13. Test final y selección del modelo
14. Sesgos y consideraciones éticas
15. Problemas que tuvimos durante el proyecto
16. Conclusiones, limitaciones y recomendaciones
17. Reproducibilidad
18. Referencias

---

## 1. Problema de negocio

La tasación de una vivienda depende del juicio de un perito, y ese proceso tiene tres problemas. Es lento, porque cada tasación requiere una visita y un informe manual. Es inconsistente, porque dos peritos pueden llegar a valores distintos para la misma casa. Y deja poca trazabilidad, porque el criterio detrás del valor rara vez queda escrito.

Para corredoras, bancos hipotecarios y plataformas de compraventa esto tiene costos concretos: casas sobrevaloradas que pasan meses sin venderse, casas subvaloradas que significan una pérdida para el vendedor y créditos hipotecarios respaldados por garantías mal estimadas.

El proyecto estima el precio de venta de una vivienda a partir de sus características físicas, de calidad y de ubicación, como apoyo a la tasación profesional. Es un problema de regresión supervisada: la variable objetivo es `SalePrice`, el precio transado en dólares. El modelo estima el valor de mercado, entendido como el precio en una venta normal entre partes independientes.

## 2. Objetivos

Objetivo general: construir un modelo que estime el precio de una vivienda residencial con un error acotado y parejo entre barrios, y que permita justificar cada estimación.

Objetivos específicos de esta entrega:

1. Ajustar los hiperparámetros de los cuatro modelos de la EP2 con validación cruzada y sin fuga de datos, y ver si los modelos de árboles dejan de sobreajustar.
2. Medir qué aportan las técnicas de ensamble: bagging, boosting, votación y stacking.
3. Evaluar la capacidad de generalización con más de una medición: folds nuevos, validación cruzada anidada, curvas de aprendizaje y un test que se abre una sola vez.
4. Elegir el modelo final con una regla fijada antes de ver los resultados y traducirlo a una herramienta de uso para el tasador.

El criterio de salida viene de la EP1 y no cambió. Un modelo es apto solo si cumple los seis indicadores a la vez sobre el conjunto de test:

| KPI | Umbral | Qué controla |
|---|---|---|
| R² | ≥ 0,85 | Qué parte de la variación de los precios explica el modelo |
| RMSE | ≤ 30.000 USD | Error en dólares, castigando más los errores grandes |
| MAE | ≤ 20.000 USD | Error promedio en dólares |
| MAPE | ≤ 12 % | Error relativo al precio de cada casa |
| Brecha de R² entre train y test | ≤ 5 puntos | Sobreajuste |
| Brecha de MAPE entre el barrio mejor y el peor predicho | ≤ 8 puntos | Equidad territorial |

Metodología. Seguimos CRISP-DM:

| Fase | Qué se hizo | Dónde |
|---|---|---|
| Comprensión del negocio | Problema, objetivos, KPI y criterio de salida | EP1; secciones 1 y 2 de este informe |
| Comprensión de los datos | Distribuciones, correlaciones, nulos, outliers, multicolinealidad | EP1; notebook, secciones 1 y 2 |
| Preparación de los datos | Limpieza, transformaciones y codificación por familia de modelos | Notebook, parte 1 (secciones 1 a 18) |
| Modelado | Cuatro modelos con parámetros iniciales (EP2), búsqueda de hiperparámetros (EP3) y ensambles (EP3) | Notebook, partes 2, 3 y 4 |
| Evaluación | Generalización, test, selección, error por segmento y uso para el tasador | Notebook, parte 4 (secciones 46 a 52) |
| Despliegue | Fuera del alcance. Queda el modelo guardado y una función de tasación | `models/modelo_final.joblib` |

## 3. Fuentes de datos

Usamos el Ames, Iowa Housing Dataset, compilado por Dean De Cock (Truman State University, 2011) a partir de los registros de la oficina de tasación de la ciudad de Ames. Es un dataset público de uso educativo.

| | |
|---|---|
| Registros | 2.930 ventas de viviendas |
| Columnas | 82: 20 continuas, 14 discretas, 23 ordinales, 23 nominales y 2 identificadores (`Order`, `PID`) |
| Período de venta | 2006 a 2010 |
| Año de construcción | 1872 a 2010 |
| Barrios | 28 |
| `SalePrice` | mediana de 160.000 USD, media de 180.796 USD, mínimo de 12.789 USD y máximo de 755.000 USD |

| Archivo | Contenido |
|---|---|
| `data/raw/Ames_Iowa_Housing_Dataset.csv` | Datos originales |
| `data/processed/ames_clean.csv` | Salida de la EP1: 2.925 filas, 80 columnas, sin nulos. Es el punto de partida del notebook |
| `data/processed/df_ridge_lasso_*.csv`, `df_random_forest_*.csv`, `df_xgboost_*.csv` | Conjuntos codificados por modelo (`train`, `test` y `test_venta_normal`), que escribe la parte 1 y lee la parte 2 |
| `data/processed/segmentos.csv` | Zona, barrio y condición de venta de cada casa, para reportar el error por segmento |

## 4. Análisis exploratorio

El análisis completo está en el informe de la EP1. Aquí resumimos lo que condicionó las decisiones del modelo.

### 4.1 Variable objetivo

Después de la limpieza de la EP1 (2.925 casas), `SalePrice` tiene una media de 180.412 USD, una mediana de 160.000 USD, una desviación estándar de 78.555 USD y una asimetría de 1,59. Unas pocas casas caras alargan la cola derecha. Por eso los modelos se entrenan sobre `log1p(SalePrice)`: la escala logarítmica corrige la asimetría y, como mostró la EP1, reparte el error de forma más pareja entre barrios.

### 4.2 Variables más relacionadas con el precio

| Variable | Correlación con `SalePrice` (EP1) |
|---|---|
| `Overall Qual` | 0,799 |
| `Gr Liv Area` | 0,707 |
| `Garage Cars` | 0,648 |
| `Total Bsmt SF` | 0,632 |
| `Year Built` | 0,558 |

El predictor más fuerte es una nota de calidad que asigna el tasador municipal, no una medida física. El modelo hereda en parte el criterio humano que busca apoyar.

### 4.3 Calidad de los datos

- 27 columnas tenían nulos, casi todos estructurales: significan que la casa no tiene la característica (sin piscina, sin garaje, sin sótano). Solo `Lot Frontage` tenía datos realmente faltantes (490 casas, 16,7 %), y se imputó con la mediana de su barrio.
- No hay filas duplicadas.
- Había cuatro pares de predictores con correlación mayor a 0,8 entre sí (por ejemplo, `Garage Cars` y `Garage Area`, 0,890). La EP1 eliminó uno de cada par.

### 4.4 Outliers

El IQR y el z-score coinciden en marcar 232 casas (7,9 %) como outlier en al menos una variable continua. El IQR marca más casos en casi todas las variables (202 frente a 28 en `Lot Frontage`). Revisamos de qué tipo era cada outlier antes de decidir:

- Artefacto del método. Cuando muchas casas tienen 0 en una variable, el IQR mezcla las casas sin la característica con las que sí la tienen. Calculado solo sobre los valores distintos de cero, el conteo baja de 200 a 63 en `Mas Vnr Area` (60,5 % de ceros) y de 156 a 86 en `Open Porch SF`.
- Cola larga natural. `Lot Area` tiene una asimetría de 13,2.
- Casos reales. Los lotes más grandes son ventas normales en terrenos bajos o con pendiente, no errores de registro.

### 4.5 Dónde se concentra el error

Con las predicciones fuera de fold de un Ridge de diagnóstico (sección 5 del notebook):

| Grupo | Casas | MAPE |
|---|---|---|
| Ventas `Normal` | 1.927 | 7,2 % |
| Ventas `Partial` (casas nuevas) | 187 | 8,9 % |
| Ventas `Family` | 41 | 16,5 % |
| Ventas `Alloca` | 20 | 17,2 % |
| Ventas `Abnorml` (remates, ventas cortas) | 155 | 19,1 % |
| Zonas residenciales | 2.314 | 8,1 % |
| Zonas no residenciales | 26 | 33,6 % |

Hay 29 casas en zona comercial, industrial o agrícola, con medianas de precio entre 47.300 y 80.313 USD, y 25 de ellas están en el barrio `IDOTRR`. Estos dos hallazgos llevaron a las pruebas de las secciones 5.5 y 5.7 de este informe.

## 5. Preparación y transformación de los datos

La preparación está en la parte 1 del notebook (secciones 1 a 18). Probamos cada decisión con validación cruzada de 5 folds solo sobre train, con un criterio escrito antes de ver el resultado. El test no participó en ninguna decisión.

### 5.1 Limpieza heredada de la EP1

- Se eliminaron 5 casas con más de 4.000 pies² de superficie habitable, como recomienda De Cock: tres eran ventas de casas en construcción con precios muy bajos para su tamaño.
- Los nulos estructurales pasaron a la categoría `None`.
- Se crearon `Tiene_Piscina`, `Tiene_Sotano`, `Tiene_Garage` y `Tiene_Chimenea`.
- Se eliminaron `Order` y `PID` (identificadores) y `Garage Area`, `Garage Yr Blt`, `TotRms AbvGrd` y `1st Flr SF` (redundantes).

### 5.2 Split

`train_test_split` con 20 % para test y semilla 42: 2.340 casas en train y 585 en test. Después de excluir las zonas no residenciales (sección 5.5) quedan 2.314 y 582. Las mismas casas se usan en las cuatro partes del notebook.

### 5.3 Outliers y `log1p`

No eliminamos ninguna fila: las casas marcadas son reales y en varias variables el IQR exageraba por los ceros.

La asimetría se trató con `log1p`. El criterio fue aplicarlo solo si reduce la asimetría al menos a la mitad sin que la correlación con el precio caiga más de 0,10. Cumplen 5 variables: `Lot Area` (asimetría de 13,2 a −0,5), `Mas Vnr Area`, `Open Porch SF`, `Wood Deck SF` y `Gr Liv Area`.

En Ridge y Lasso comparamos cinco variantes:

| Variante | Tratamiento | Resultado |
|---|---|---|
| A | Escalado de la EP1 (`RobustScaler` y `StandardScaler`) | Base |
| B | A + `log1p` en las 5 variables | MAPE unos 0,1 puntos menor; se adopta |
| C | B + recorte al percentil 99 de otras 5 superficies | RMSE unos 270 USD mayor; se descarta |
| D | C sin zonas no residenciales | No comparable con las demás; se decide aparte (sección 5.5) |
| E | B + `StandardScaler` en las ordinales | Igual que B; se adopta para que la penalización trate a todas las variables en la misma escala |

### 5.4 Qué necesitan los modelos de árboles

- Un árbol solo usa el orden de los valores para cortar, así que `log1p` y el escalado no deberían cambiarlo. Lo comprobamos: 3 USD de diferencia de RMSE en Random Forest y resultados idénticos en XGBoost.
- En Random Forest, las cuatro codificaciones de las categóricas que probamos quedan a menos de 100 USD entre sí, frente a una variación entre folds de unos 2.500 USD.
- En XGBoost, one-hot con target encoding baja el RMSE en unos 670 USD (2,9 %) respecto de la codificación nativa de categorías.

Los dos usan la misma codificación: one-hot para las nominales de baja cardinalidad y target encoding para `Neighborhood`, `Exterior 1st` y `Exterior 2nd`.

### 5.5 Zonas no residenciales

Comparar el error con y sin estas casas no sirve, porque al sacarlas también se dejan de evaluar. En cada fold entrenamos dos modelos, con y sin las no residenciales, y los evaluamos sobre las mismas casas residenciales de validación.

Criterio: las no residenciales empeoran a un modelo si el RMSE sobre residenciales sube en promedio y en al menos 4 de 5 folds; se excluyen si eso pasa en la mayoría de los modelos. Empeoran a Ridge, Lasso y XGBoost (entre 70 y 190 USD de RMSE), y el modelo tampoco las predice bien (MAPE entre 34 % y 44 %). Se excluyen como criterio de alcance: el modelo tasa viviendas en zonas residenciales y el resto va a tasación manual. Es un criterio aplicable a una casa nueva, porque la zonificación se conoce antes de tasar.

### 5.6 Variables casi constantes y redundantes

Probamos un paquete de limpieza: quitar `Utilities` y `Street` (casi todas las casas tienen el mismo valor), `Pool QC` y `Pool Area` (solo 11 piscinas; queda `Tiene_Piscina`) y `BsmtFin SF 2` (correlación de 0,008 con el precio y 88 % de ceros), y agrupar las categorías con menos de 10 casas en train. Criterio: se adopta salvo que empeore el RMSE en promedio y en al menos 4 de 5 folds. Ridge mejora 66 USD, Lasso 221 USD y ningún modelo de árboles empeora según el criterio. Se adopta en los tres conjuntos.

### 5.7 Variables que solo se conocen al vender

`Sale Condition` y `Sale Type` se conocen al vender, no al tasar. Las mantuvimos como variables de control: quitarlas empeora a los cuatro modelos (el RMSE sube 505 USD en Ridge, 481 en Lasso, 609 en XGBoost y 20 en Random Forest). Al tasar una casa nueva se asume una venta normal (`Normal` / `WD`) en lo que no se conoce de antemano; las casas nuevas (`Partial` / `New`) sí se conocen antes y conservan su valor. Para medir ese escenario preparamos un segundo test con la venta normal impuesta, en el que cambian 64 de las 582 casas. También se fija `Yr Sold = 2010` al tasar, para que el modelo no extrapole fuera del período de los datos.

### 5.8 Resumen

| | Ridge / Lasso | Random Forest / XGBoost |
|---|---|---|
| Filas eliminadas por outlier | Ninguna | Ninguna |
| Zonas no residenciales | Excluidas | Excluidas |
| `log1p` en 5 variables de X | Sí | No |
| Escalado | `RobustScaler` y `StandardScaler` | No |
| Ordinales | Mapeo manual + `StandardScaler` | Mapeo manual |
| Nominales de baja cardinalidad | One-hot (`drop='first'`, `min_frequency=10`) | One-hot (`min_frequency=10`) |
| `Neighborhood`, `Exterior 1st`, `Exterior 2nd` | Target encoding + `StandardScaler` | Target encoding |
| Columnas eliminadas | `Utilities`, `Street`, `Pool QC`, `Pool Area`, `BsmtFin SF 2` | Igual |
| Variables de la venta | Se mantienen | Se mantienen |
| Target | `log1p(SalePrice)` al entrenar | Igual |

Todo lo que aprende de los datos (escalado y target encoding) va dentro de un `Pipeline`. El target encoding usa el `TargetEncoder` de scikit-learn, que calcula la codificación de cada casa de train sin su propio precio (cross-fitting), así que no hay fuga del target.

## 6. Selección de modelos

Comparamos cuatro modelos de las tres familias que preseleccionó la EP1, de menor a mayor complejidad, y en la parte 4 dos combinaciones de ellos.

| Familia | Modelo | Por qué se incluye | Limitación |
|---|---|---|---|
| Regresión lineal regularizada | Ridge | Línea base interpretable: cada estimación se descompone en el aporte de cada variable. La penalización L2 controla los coeficientes con variables correlacionadas | Supone relaciones lineales (en escala log) |
| Regresión lineal regularizada | Lasso | Igual que Ridge, pero la penalización L1 deja coeficientes en 0: selecciona variables y deja un modelo más simple | Entre variables muy correlacionadas tiende a quedarse con una sola |
| Ensamble por bagging | Random Forest | Promedia cientos de árboles entrenados con muestras distintas; capta relaciones no lineales sin especificarlas | Menos interpretable; con árboles profundos puede sobreajustar |
| Ensamble por boosting | XGBoost | Suma árboles chicos, cada uno ajustado al error de los anteriores | Muchos hiperparámetros y menos transparencia |
| Combinación de modelos | Votación y stacking de Lasso con XGBoost | Combinan un lineal y un ensamble de árboles, que se equivocan en casas distintas | Heredan la opacidad del modelo de árboles |

Si un modelo lineal rinde parecido a los demás, se prefiere, porque el problema de negocio pide poder justificar el valor ante un cliente o un banco. Esa preferencia está escrita en la regla de elección (sección 9.1) desde la EP2.

## 7. Variables finales por modelo

Los dos conjuntos parten de las mismas 74 variables originales: las 79 predictoras de `ames_clean.csv` menos las 5 eliminadas en la sección 5.6. Cambia la forma de codificarlas.

| Grupo | Variables | Ridge / Lasso | Random Forest / XGBoost |
|---|---|---|---|
| Superficies con `log1p` | `Lot Area`, `Mas Vnr Area`, `Open Porch SF`, `Wood Deck SF`, `Gr Liv Area` | 5 columnas, `log1p` + `StandardScaler` | 5 columnas, sin transformar |
| Otras superficies | `Lot Frontage`, `BsmtFin SF 1`, `Bsmt Unf SF`, `Total Bsmt SF`, `2nd Flr SF` | 5 columnas, `RobustScaler` | 5 columnas, sin transformar |
| Otras numéricas | Años, notas generales (`Overall Qual`, `Overall Cond`), baños, dormitorios, cocinas, chimeneas, `Garage Cars`, porches, `Low Qual Fin SF`, `Misc Val` | 19 columnas, `StandardScaler` | 19 columnas, sin transformar |
| Ordinales | 18 variables de calidad, condición, exposición, terminación, funcionalidad, forma y pendiente del lote | 18 columnas, mapeo manual + `StandardScaler` | 18 columnas, mapeo manual |
| Nominales de alta cardinalidad | `Neighborhood`, `Exterior 1st`, `Exterior 2nd` | 3 columnas, target encoding + `StandardScaler` | 3 columnas, target encoding |
| Nominales de baja cardinalidad | 20 variables, entre ellas `MS SubClass`, `MS Zoning`, `House Style`, `Foundation`, `Sale Type` y `Sale Condition` | 84 dummies (sin la primera categoría de cada variable) | 104 dummies |
| Binarias | `Tiene_Piscina`, `Tiene_Sotano`, `Tiene_Garage`, `Tiene_Chimenea` | 4 columnas | 4 columnas |
| Total | 74 variables | 138 columnas | 158 columnas |

En los lineales se elimina la primera dummy de cada variable porque, con intercepto, las dummies de una misma variable suman siempre 1 y generan colinealidad perfecta. Los árboles no tienen ese problema.

El modelo final (Lasso con `alpha` = 0,000174) deja en 0 los coeficientes de 33 columnas y usa 105 de las 138. Con el `alpha` que eligió la búsqueda de la EP3 (0,000127) usaría 111.

## 8. Implementación

### 8.1 Herramientas

Python con pandas y numpy para los datos, scikit-learn para la preparación, los modelos lineales, Random Forest, los ensambles y la validación, y XGBoost para el boosting. Versiones fijas en `requirements.txt`: numpy 2.1.3, pandas 2.2.3, scikit-learn 1.6.1 y xgboost 3.4.1, con Python 3.13. Las cifras de este informe salen de la ejecución completa del notebook en Windows del 20-11-2026, salvo las de la sección 15.3, que salen de ejecutar el mismo notebook en Linux.

### 8.2 Estructura del notebook

Un solo notebook, en cuatro partes:

| Parte | Secciones | Qué hace |
|---|---|---|
| 1 | 1 a 18 | Prepara los datos y escribe los CSV de `data/processed/` |
| 2 | 19 a 27 | Lee esos CSV, compara los cuatro modelos con parámetros iniciales y elige Lasso (EP2) |
| 3 | 28 a 40 | Busca los hiperparámetros de los cuatro modelos y guarda la evidencia en `models/busquedas/` y `models/mejores_params.json` |
| 4 | 41 a 52 | Ensambles, generalización, test final, selección, función de tasación y modelo final guardado |

Decisiones de implementación que se repiten en todo el notebook:

- Cada modelo va en un `Pipeline` con un `ColumnTransformer` que aplica a cada grupo de columnas su transformación, y todo dentro de un `TransformedTargetRegressor` que entrena sobre `log1p(SalePrice)` y devuelve dólares con `expm1`. Así las métricas quedan en dólares.
- Validación cruzada de 5 folds (`KFold` mezclado, semilla 42) sobre las 2.314 casas de train, con las mismas particiones en todo el proyecto.
- En las partes 3 y 4 el paralelismo va en un solo nivel: la búsqueda o `cross_validate` reparte el trabajo entre los núcleos y cada modelo usa un hilo.
- Los cálculos largos se guardan en JSON y se cargan si ya existen. Los interruptores `REPETIR_BUSQUEDAS` (parte 3) y `REPETIR_PARTE4` (parte 4) obligan a recalcularlos.
- La celda del test (sección 48) deja una bandera, `TEST_ABIERTO`, que impide ejecutarla dos veces en la misma sesión.

## 9. Modelos con parámetros iniciales (EP2)

### 9.1 Regla de elección

Fijada en la EP2, antes de entrenar, y usada sin cambios en la EP3:

1. Solo compiten los modelos que cumplen los seis KPI en el test real.
2. Entre ellos se prefiere el de menor RMSE en validación cruzada (la métrica principal de la EP1). Se elige por validación y no por test, para que el test quede como una medición independiente.
3. Si otro modelo que cumple queda a menos de 1 punto de R² de validación, se elige el más interpretable: primero los lineales y, entre ellos, el que usa menos columnas.
4. Si ningún modelo cumple, no se bajan los umbrales: se vuelve a una fase anterior.

### 9.2 Resultados

Validación cruzada sobre los CSV codificados (sección 22):

| Modelo | RMSE train | RMSE val. | Desv. entre folds | MAPE val. | R² train | R² val. | Brecha |
|---|---|---|---|---|---|---|---|
| Ridge | 18.831 | 20.152 | 1.018 | 8,07 % | 0,940 | 0,930 | 1,0 |
| Lasso | 18.919 | 20.206 | 1.077 | 8,08 % | 0,939 | 0,930 | 0,9 |
| XGBoost | 3.598 | 22.529 | 1.554 | 8,40 % | 0,998 | 0,913 | 8,5 |
| Random Forest | 9.475 | 25.137 | 2.505 | 9,23 % | 0,985 | 0,891 | 9,3 |

Test real (sección 23):

| Modelo | R² | RMSE | MAE | MAPE | Brecha train-test | Brecha barrios | Cumple los seis |
|---|---|---|---|---|---|---|---|
| Ridge | 0,948 | 19.126 | 12.373 | 6,75 % | −0,9 | 6,1 | Sí |
| Lasso | 0,948 | 19.241 | 12.407 | 6,77 % | −0,9 | 6,1 | Sí |
| XGBoost | 0,934 | 21.660 | 13.579 | 7,39 % | 6,3 | 7,7 | No |
| Random Forest | 0,924 | 23.197 | 14.312 | 7,92 % | 6,1 | 10,0 | No |

Random Forest y XGBoost casi memorizaban train y no cumplían el KPI de sobreajuste. Ridge tenía el menor RMSE de validación, pero Lasso quedaba a 0,04 puntos de R² con 105 columnas en vez de 138, y por la regla 3 la EP2 eligió Lasso. La EP2 dejó dos pendientes: ajustar los hiperparámetros con pipelines completos (la validación sobre los CSV tenía una fuga leve, porque la preparación se había ajustado con todo train) y ver si así los árboles dejaban de sobreajustar.

## 10. Optimización de hiperparámetros y validación cruzada

Es la parte 3 del notebook (secciones 28 a 40).

### 10.1 Protocolo

Lo escribimos antes de ejecutar la primera búsqueda (sección 28):

- Mismos datos, folds y métricas que la EP2. Cada configuración se evalúa con el pipeline completo dentro de la validación cruzada, sobre las casas sin codificar, para que el escalado y el target encoding se ajusten de nuevo en cada fold. Así se elimina la fuga de la EP2.
- El test no se usa en ninguna celda de esta parte.
- Regla para elegir la configuración de cada modelo: entre las que tienen una brecha de R² entre train y validación de hasta 5 puntos, la de menor RMSE de validación. El KPI de sobreajuste es eliminatorio, así que una configuración que baja el error memorizando train no sirve. Es un criterio conservador: en la EP2 la brecha de los árboles fue mayor en validación (9,3 y 8,5) que en test (6,1 y 6,3).
- Cuándo un cambio es real: si mejora en promedio, en al menos 4 de los 5 folds y por más que la desviación del RMSE entre folds. Una diferencia menor es ruido de las particiones.

### 10.2 Línea base comparable

Volvimos a medir los cuatro modelos con los parámetros de la EP2, ahora con pipelines completos (sección 29): Ridge 20.159 USD, Lasso 20.199, Random Forest 25.151 y XGBoost 22.351. La diferencia con la validación sobre los CSV es de 8, −7, 14 y −178 USD: la fuga de la EP2 era chica. Esta tabla es el "antes" de la comparación.

### 10.3 Método

Para Random Forest y XGBoost seguimos cuatro pasos:

1. Diagnóstico: desde la configuración de la EP2, mover un hiperparámetro a la vez para ver cuáles bajan la brecha y a qué costo (sección 30.2). De ahí salen los rangos de búsqueda.
2. `RandomizedSearchCV` con 60 configuraciones sorteadas. Con varios hiperparámetros, una grilla gasta casi todos sus ajustes en repetir los mismos valores de cada uno, y normalmente solo unos pocos importan (Bergstra y Bengio, 2012). El número 60 sale de 1 − 0,95^60 = 0,954: con 60 sorteos hay un 95 % de probabilidad de que al menos uno caiga en el 5 % mejor del espacio.
3. `GridSearchCV` fino en la zona de la mejor aleatoria y, si la mejor quedaba en el borde de un rango, una grilla ampliada.
4. Curvas de validación alrededor de la configuración elegida, como evidencia de cada valor y como control. No las usamos para seguir afinando (ver sección 15.6).

En Ridge y Lasso solo se busca `alpha`, con un `GridSearchCV` sobre las mismas 30 alternativas que usaron `RidgeCV` y `LassoCV` en la EP2.

Medimos el costo antes de buscar: un ajuste tomaba 6,9 segundos en Random Forest y 1,1 en XGBoost en un equipo de 12 núcleos, así que 60 configuraciones por 5 folds eran unos 3 minutos y medio minuto. No hizo falta reducir el número de configuraciones.

### 10.4 Espacios de búsqueda

| Modelo | Hiperparámetro | EP2 | Rango | Por qué |
|---|---|---|---|---|
| Random Forest | `n_estimators` | 300 | Fijo en 300 | Entre 300 y 800 árboles el RMSE cambia 42 USD y la brecha no se mueve. Más árboles no sobreajustan (Breiman, 2001) |
| Random Forest | `max_features` | 1,0 | 0,1 a 1,0 | El mínimo del diagnóstico está en 0,4 |
| Random Forest | `min_samples_leaf` | 1 | 1 a 20 | Por sí solo cumple la brecha desde 12 casas por hoja |
| Random Forest | `max_depth` | Sin límite | 4 a 20 y sin límite | Bajo 4 el error se dispara; desde 14 es igual a no limitar |
| Random Forest | `max_samples` | 1,0 | 0,1 a 1,0 | Por sí solo cumple la brecha con 0,3 |
| XGBoost | `max_depth` | 6 | 1 a 6 | El mínimo del diagnóstico está entre 2 y 4 |
| XGBoost | `n_estimators` | 500 | 100 a 2.000 | Con tasas bajas se necesitan más árboles |
| XGBoost | `learning_rate` | 0,05 | 0,01 a 0,2 | El mínimo del diagnóstico está en 0,05 (Friedman, 2001) |
| XGBoost | `min_child_weight` | 1 | 1 a 40 | Hasta 40 la brecha baja sin subir el error |
| XGBoost | `subsample` | 1,0 | 0,3 a 1,0 | El mejor valor del diagnóstico fue el menor probado (Friedman, 2002) |
| XGBoost | `colsample_bytree` | 1,0 | 0,1 a 1,0 | El mínimo del diagnóstico está en 0,3 |
| XGBoost | `reg_lambda` | 1 | 0 a 100 | Penalización L2 de las hojas (Chen y Guestrin, 2016) |
| XGBoost | `reg_alpha` | 0 | 0 a 1 | Penalización L1 de las hojas, en la escala del log del precio |
| XGBoost | `gamma` | 0 | 0 a 0,1 | Ganancia mínima de un corte, en la escala del log del precio |
| Ridge | `alpha` | 8,53 | 30 valores entre 0,01 y 1.000 | La grilla de la EP2 |
| Lasso | `alpha` | 0,000174 | 30 valores entre 0,00001 y 0,1 | La grilla de la EP2 |

Random Forest tiene 8.800 combinaciones posibles y XGBoost 121.927.680. El rango de `gamma` salió de medir la ganancia de los cortes del XGBoost de la EP2: como el target está en log, la mediana es 0,0049 y el 98 % de los cortes gana menos de 1. Con `gamma` = 1, ese 98 % de los cortes no se haría; por eso el rango llega solo hasta 0,1.

### 10.5 Resultados de la búsqueda

Random Forest (sección 31): de las 60 configuraciones aleatorias, 48 cumplen la brecha y la mejor tiene 25.668 USD. La grilla fina (240 configuraciones) baja a 25.512 USD con `max_features` 0,4, hojas de 5 casas, `max_samples` 0,7 y sin límite de profundidad, con brecha de 4,8. La grilla ampliada (64) no la mejora. En total, 359 configuraciones.

XGBoost (sección 32): de las 60 aleatorias, 42 cumplen y la mejor tiene 20.348 USD. La grilla fina (270) baja a 19.809 USD con 2.000 árboles de profundidad 2, tasa 0,05, `subsample` 0,7, `colsample_bytree` 0,5, `min_child_weight` 8, `reg_lambda` 10 y `gamma` y `reg_alpha` en 0,01, con brecha de 3,9. La grilla ampliada (144) no la mejora. En total, 467 configuraciones.

Ridge y Lasso (sección 36): Ridge elige `alpha` = 3,86 (la EP2, 8,53) y Lasso 0,000127 (la EP2, 0,000174). Los dos quedan al interior de la grilla y su error casi no cambia.

### 10.6 Evidencia de cada valor

- Curvas de validación (sección 33). Ningún valor vecino cumple el criterio de cambio real. En XGBoost, `min_child_weight` 1 mejora 107 USD y 3.000 árboles mejoran 58, los dos en 4 de 5 folds, pero la desviación entre folds es 570 USD. Con profundidad 1 el error sube a 20.682 USD y con profundidad 3 la brecha pasa de 5. En Random Forest, `min_samples_leaf` es donde mejor se ve el sobreajuste: con hojas de 1 casa el RMSE de train baja de 19.246 a 13.191 USD y el de validación solo de 25.512 a 24.735, con brecha de 7,5. El valor elegido, 5, es el menor que cumple (con 3 la brecha es 6,11).
- Qué hiperparámetros importan (sección 34). En Random Forest deciden `min_samples_leaf` y `max_samples`, por su efecto en la brecha. En XGBoost, `n_estimators` tiene el mayor efecto en su rango (7.240 USD), pero alrededor del valor elegido solo `max_depth` mueve el error más que la desviación entre folds (874 USD).
- Suficiencia de la búsqueda (sección 35). En Random Forest las grillas bajaron la mejor aleatoria en 156 USD (0,05 desviaciones entre folds) y en XGBoost en 540 USD (0,95). En XGBoost la última mejora apareció en la configuración 195 de 467. Hay una zona plana: 134 de las 399 configuraciones de XGBoost que cumplen la brecha quedan a menos de una desviación entre folds de la elegida.
- Costo de exigir la brecha (sección 35.3). En Random Forest la mejor configuración sin restricción tiene 24.735 USD con 7,5 puntos de brecha: exigir la brecha cuesta 777 USD. En XGBoost no cuesta nada, porque la mejor configuración ya cumple.

### 10.7 Antes y después

Validación cruzada de 5 folds sobre train, pipelines completos (sección 37):

| Modelo | RMSE val. EP2 | RMSE val. EP3 | Brecha EP2 | Brecha EP3 | Folds en que mejora | ¿Mejora real? |
|---|---|---|---|---|---|---|
| Ridge | 20.159 | 20.123 | 1,0 | 1,1 | 3 de 5 | No |
| Lasso | 20.199 | 20.171 | 1,0 | 1,1 | 2 de 5 | No |
| Random Forest | 25.151 | 25.512 | 9,1 | 4,8 | 1 de 5 | No |
| XGBoost | 22.351 | 19.809 | 8,2 | 3,9 | 5 de 5 | Sí |

- XGBoost es el único con una mejora real: baja 2.542 USD, en los 5 folds, y la brecha pasa de 8,2 a 3,9 puntos. El cambio que más pesa es la profundidad: de 6 a 2, con más árboles y submuestreo de casas.
- Random Forest pasa a cumplir la brecha, pero su error sube 362 USD. Dejó de memorizar train sin predecir mejor las casas nuevas.
- Ridge y Lasso no cambian: su `alpha` ya estaba bien elegido en la EP2.
- XGBoost supera a Lasso por 0,27 puntos de R² de validación y le gana en 3 de 5 folds, lejos del punto que exige la regla para preferir un modelo menos interpretable.

Configuración final de cada modelo:

| Modelo | Hiperparámetros | RMSE val. | Brecha |
|---|---|---|---|
| Ridge | `alpha` 3,86 | 20.123 USD | 1,1 |
| Lasso | `alpha` 0,000127 | 20.171 USD | 1,1 |
| Random Forest | 300 árboles, `max_features` 0,4, `min_samples_leaf` 5, `max_samples` 0,7, sin límite de profundidad | 25.512 USD | 4,8 |
| XGBoost | 2.000 árboles de profundidad 2, tasa 0,05, `subsample` 0,7, `colsample_bytree` 0,5, `min_child_weight` 8, `reg_lambda` 10, `reg_alpha` 0,01, `gamma` 0,01 | 19.809 USD | 3,9 |

Estos parámetros quedan en `models/mejores_params.json`, que la parte 4 lee con `construir_modelo_p3(nombre)`.

## 11. Técnicas de ensamble

Es la parte 4 del notebook (secciones 41 a 45).

### 11.1 Bagging y boosting

Medimos cada mecanismo con la configuración de la EP3 fija y los mismos 5 folds (sección 43):

- Bagging (Random Forest). Un solo árbol del bosque tiene 37.604 USD de RMSE de validación y 8,6 puntos de brecha. Con 25 árboles promediados baja a 25.965 y con 300 a 25.512, con brecha de 4,8. El error de train casi no baja después de 25 árboles (de 19.691 a 19.246): lo que cae es la varianza.
- Boosting (XGBoost). Con 1 ronda el modelo predice casi la media (75.971 USD). El error baja a 20.892 con 500 rondas y se aplana desde 1.000 (19.988; 19.822 con 1.500). Cada árbol de profundidad 2 es pobre; la suma corrige el sesgo.

El árbol de 1 elemento es un árbol del bosque, que usa el 40 % de las columnas en cada corte y el 70 % de las casas. No ajustamos por separado un árbol de decisión completo, así que la ganancia del bagging frente a un árbol bien ajustado es probablemente menor que los 12.092 USD de esta curva.

### 11.2 Votación y stacking

Antes de combinar medimos si Lasso y XGBoost se equivocan en casas distintas (sección 44.1). La correlación entre sus residuos fuera de fold es 0,874: parecidos, pero no iguales. En 76 casas Lasso se equivoca más de 10 % y XGBoost menos de 5 %, y en 59 casas pasa lo contrario.

Probamos dos combinaciones, decididas antes de ver resultados (sección 44.2). Las bases se combinan en escala log y cada una trae su propia preparación:

| Tipo | Modelos | RMSE val. | Desv. entre folds | Brecha | Gana a XGBoost | Gana a Lasso |
|---|---|---|---|---|---|---|
| Votación (promedio simple) | Lasso + XGBoost | 19.239 | 787 | 2,5 | 5 de 5 | 5 de 5 |
| Votación | Lasso + RF + XGBoost | 20.411 | 1.450 | 3,0 | 2 de 5 | 3 de 5 |
| Stacking (regresión lineal con coeficientes positivos) | Lasso + XGBoost | 19.203 | 768 | 2,5 | 5 de 5 | 5 de 5 |
| Stacking | Lasso + RF + XGBoost | 19.225 | 822 | 2,5 | 5 de 5 | 5 de 5 |

Agregar Random Forest no sirve. En la votación empeora el resultado, y en el stacking el meta-modelo le da un peso de 3 %. En el stacking de Lasso y XGBoost los pesos son 45 % y 55 %, muy cerca del 50/50 de la votación. Pasan al test la votación y el stacking de Lasso con XGBoost.

### 11.3 ¿Los ensambles mejoran?

Con el criterio de cambio real (sección 45):

| Comparación | Diferencia de RMSE val. | Desv. de la referencia | Folds en que gana | Diferencia de R² | ¿Cambio real? |
|---|---|---|---|---|---|
| Votación vs XGBoost | −570 | 570 | 5 de 5 | +0,38 | No (queda en el límite) |
| Stacking vs XGBoost | −605 | 570 | 5 de 5 | +0,40 | Sí |
| Random Forest vs Lasso | +5.342 | 976 | 0 de 5 | −4,11 | No |
| XGBoost vs Lasso | −362 | 976 | 3 de 5 | +0,27 | No |
| Votación vs Lasso | −932 | 976 | 5 de 5 | +0,64 | No |
| Stacking vs Lasso | −967 | 976 | 5 de 5 | +0,66 | No |

Los ensambles sí mejoran a los árboles: bagging y boosting los vuelven competitivos, y combinar Lasso con XGBoost da el menor error medido. Frente a Lasso, ninguno logra un cambio real: el stacking gana en los 5 folds, pero por menos que la variación entre folds y por 0,66 puntos de R², bajo el punto que exige la regla.

## 12. Capacidad de generalización

### 12.1 Folds nuevos

Cada configuración se eligió como la mejor entre cientos evaluadas con los mismos 5 folds, así que parte de su ventaja puede ser suerte con esas particiones. Reevaluamos los siete candidatos con 15 folds que la búsqueda nunca vio (`RepeatedKFold`, 3 repeticiones de 5 folds, semilla 2024; sección 46.1):

| Modelo | Configuraciones probadas | RMSE val. folds de la búsqueda | RMSE val. folds nuevos | Diferencia |
|---|---|---|---|---|
| Ridge | 30 | 20.123 | 20.160 | +37 (0,2 %) |
| Lasso | 30 | 20.171 | 20.224 | +53 (0,3 %) |
| Lasso (`alpha` de la EP2) | 30 | 20.174 | 20.243 | +69 (0,3 %) |
| Random Forest | 359 | 25.512 | 25.502 | −10 (0,0 %) |
| XGBoost | 467 | 19.809 | 20.248 | +440 (2,2 %) |
| Votación | 2 | 19.239 | 19.509 | +270 (1,4 %) |
| Stacking | 2 | 19.203 | 19.489 | +286 (1,5 %) |

En los lineales y en Random Forest el optimismo es despreciable. XGBoost, el modelo con más configuraciones probadas, sí muestra algo: sube 440 USD con folds nuevos y queda empatado con Lasso (gana en 7 de 15). El stacking le gana a Lasso en los 15 folds nuevos, por 735 USD.

### 12.2 Validación cruzada anidada

Mide el error del procedimiento de búsqueda completo: en cada fold externo se repite la búsqueda solo con las casas de entrenamiento de ese fold y se mide en el fold externo (sección 46.2). Para Lasso, la grilla de 30 `alpha`; para XGBoost, 20 configuraciones aleatorias por fold.

- Lasso: 20.158 USD con la búsqueda anidada, frente a 20.171 de la configuración elegida. Elegir `alpha` no introduce optimismo.
- XGBoost: 21.001 USD, 1.192 más. Parte es el optimismo de la sección 12.1 y parte es el costo de una búsqueda chica: con 20 configuraciones cada fold elige algo distinto. Es una cota pesimista del procedimiento de la EP3, que probó 467 configuraciones.

### 12.3 Curvas de aprendizaje

Entrenamos con el 10 % al 100 % de las casas de entrenamiento de cada fold (sección 47):

- Lasso converge: en el último tramo (de 1.613 a 1.851 casas) su error baja 117 USD y al final train y validación quedan a 1.458 USD.
- XGBoost aprende más rápido con pocos datos, pero al final queda una distancia de 7.266 USD entre train y validación y su error todavía baja 428 USD en el último tramo. Es el que más ganaría con más datos.
- El stacking queda entre los dos.

### 12.4 Validación frente a test

Si el modelo generaliza, el error de test tiene que quedar dentro de lo que anticipó la validación y el orden de los modelos tiene que ser el mismo (sección 48.1). El orden es idéntico en validación y en test: stacking, votación, XGBoost, Ridge, los dos Lasso y Random Forest. En todos los modelos el RMSE de test queda bajo el de validación, entre 933 y 1.047 USD en los lineales y entre 1.224 y 1.965 USD en los ensambles. Igual que en la EP2, estas 582 casas resultaron algo más fáciles que el promedio. La brecha de los árboles en test también bajó respecto de validación: XGBoost de 3,9 a 1,6 puntos y Random Forest de 4,8 a 1,7.

## 13. Test final y selección del modelo

### 13.1 Test

Los siete candidatos se definieron antes de abrir el test (sección 41) y se midieron una sola vez, entrenados con las 2.314 casas de train (sección 48). Lasso con el `alpha` de la EP2 entró a la lista porque tiene el mismo error que el de la EP3 con 6 columnas menos.

Test real:

| Modelo | R² | RMSE | MAE | MAPE | Brecha R² train-test | Brecha barrios | KPI cumplidos |
|---|---|---|---|---|---|---|---|
| Ridge | 0,949 | 19.076 | 12.340 | 6,74 % | −0,9 | 6,2 | 6 de 6 |
| Lasso | 0,948 | 19.181 | 12.383 | 6,77 % | −0,9 | 6,3 | 6 de 6 |
| Lasso (`alpha` de la EP2) | 0,948 | 19.241 | 12.407 | 6,77 % | −0,9 | 6,1 | 6 de 6 |
| Random Forest | 0,922 | 23.548 | 14.280 | 7,94 % | 1,7 | 12,2 | 5 de 6 |
| XGBoost | 0,953 | 18.241 | 11.669 | 6,46 % | 1,6 | 6,4 | 6 de 6 |
| Votación | 0,954 | 18.015 | 11.462 | 6,30 % | 0,5 | 5,7 | 6 de 6 |
| Stacking | 0,955 | 17.903 | 11.418 | 6,28 % | 0,6 | 5,7 | 6 de 6 |

- Random Forest no cumple la equidad entre barrios (12,2 puntos, peor barrio `IDOTRR` con 17,1 %), como en la EP2.
- Con la búsqueda de la EP3, XGBoost ya cumple el KPI de sobreajuste en test (1,6 puntos; en la EP2 tenía 6,3).
- Con la venta normal impuesta, XGBoost no cumple la equidad (9,3 puntos). La votación y el stacking cumplen, pero cerca del límite: 7,4 y 7,5 puntos frente a un máximo de 8. En Linux esos dos pasan a 8,5 y 8,6 y dejan de cumplir (sección 15.3).

### 13.2 Selección

La regla de la sección 9.1, aplicada en código (sección 49):

1. Cumplen los seis KPI en el test real: Ridge, los dos Lasso, XGBoost, la votación y el stacking.
2. Menor RMSE de validación entre ellos: el stacking, con 19.203 USD y R² de 0,9364.
3. A menos de 1 punto de R² del stacking quedan los seis. Gana el más interpretable: Lasso con el `alpha` de la EP2, lineal con 105 columnas en uso (Lasso con el `alpha` de la EP3 usa 111 y Ridge 138).

Como los ensambles cumplen los KPI pero no superan a Lasso por más de 1 punto de R² de validación, aplica el escenario (b) del protocolo: se mantiene un lineal por interpretabilidad.

El modelo final es Lasso con `alpha` = 0,000174, el mismo valor que eligió la EP2.

### 13.3 Por qué Lasso y no el stacking

El stacking tiene menos error en validación, en folds nuevos y en test. Elegimos Lasso por cinco razones:

1. La regla se fijó antes de ver los resultados y la aplicamos sin cambios. Cambiarla después de ver que el stacking gana sería ajustar el criterio al resultado.
2. La ventaja es consistente pero chica: 971 USD de RMSE de validación, menos que la desviación entre folds de Lasso (991 USD). En test son 1.338 USD y 0,5 puntos de MAPE, frente a un error mediano del propio Lasso de 8.659 USD.
3. Trazabilidad: Lasso explica cada tasación como la suma de aportes de cada variable. El stacking combina un lineal con 2.000 árboles, y para justificar un valor haría falta SHAP.
4. Robustez: el resultado de Lasso no depende de cómo se buscaron sus hiperparámetros (validación anidada: −13 USD) ni del sistema operativo (sección 15.3). El de XGBoost, que es la mitad del stacking, sí depende de las dos cosas.
5. Equidad con venta normal: el stacking queda a 0,5 puntos del máximo en Windows y lo pasa en Linux. Lasso queda en 6,2 en los dos sistemas.

Para reemplazar a Lasso, un ensamble tendría que superar su R² de validación por más de 1 punto (R² mayor que 0,9397) y mantener los seis KPI en los dos escenarios de test.

### 13.4 Dónde se equivoca el modelo elegido

Test real, Lasso con el `alpha` de la EP2, con el stacking al lado (sección 50):

| Segmento | MAPE Lasso | MAPE stacking |
|---|---|---|
| Cuartil de precio más bajo (hasta 130.625 USD) | 8,8 % | 8,8 % |
| 130.625 a 160.000 USD | 5,8 % | 5,0 % |
| 160.000 a 215.000 USD | 5,8 % | 5,0 % |
| Más de 215.000 USD | 6,6 % | 6,3 % |
| Ventas `Normal` (483 casas) | 6,3 % | 5,8 % |
| Ventas `Abnorml` (33 casas) | 11,2 % | 11,8 % |
| Ventas `Family` (5 casas) | 15,5 % | 13,6 % |

- En el cuartil más barato Lasso sobreestima un 3,4 % en promedio y el stacking no lo corrige.
- De las 9 casas de test de más de 450.000 USD, los dos modelos subestiman 8, en −12,5 % (Lasso) y −11,5 % (stacking) en promedio.
- Donde el stacking mejora es en el centro de la distribución y en 15 de los 17 barrios con al menos 10 casas en test.
- Con Lasso, la mitad de las casas tiene un error menor a 5,3 %, el 77,8 % menor a 10 % y el 96,6 % menor a 20 %. Solo 4 casas superan el 30 %.

El ensamble no corrige los extremos, que es donde el perito más necesita ayuda. Esas casas se derivan a revisión manual con cualquiera de los dos modelos.

### 13.5 Uso para el tasador

La función `tasar(casa)` (sección 51) recibe una casa con las 79 variables sin codificar y devuelve:

- El valor estimado, con la venta normal impuesta y `Yr Sold = 2010`.
- Un rango orientativo: el valor multiplicado por los percentiles 10 y 90 del cociente entre precio real y estimado de las predicciones fuera de fold, que va de −11,1 % a +13,5 %. No es un intervalo de confianza formal.
- El desglose del valor en el aporte de cada variable, respecto de una casa con los valores promedio de train.
- Alertas de revisión manual: valor sobre 450.000 USD, cuartil más barato, datos atípicos respecto de train y zonas no residenciales, que quedan fuera de alcance.

Ejemplos con casas reales del test:

- Casa típica de `Mitchel`, vendida en 160.000 USD: el modelo estima 175.624 USD (+9,8 %), dentro del rango (156.211 a 199.332). La superficie habitable suma 8,3 % y la condición general 6,2 %; la funcionalidad resta 7,9 % y el año de construcción 4,6 %.
- La casa más cara del test (611.657 USD, nueva, en `NridgHt`): estima 472.711 USD (−22,7 %), fuera del rango, pero la herramienta marca el valor sobre 450.000 USD y una superficie de sótano terminado atípica.
- La más barata (46.500 USD, venta `Abnorml` en `OldTown`): estima 56.235 USD (+20,9 %), con alertas de cuartil barato y superficie atípica.
- La misma casa típica en zona comercial: la herramienta no estima y deriva a tasación manual.

El modelo final queda guardado en `models/modelo_final.joblib`, con su ficha en `models/modelo_final.json`.

## 14. Sesgos y consideraciones éticas

### 14.1 Sesgos de los datos (EP1)

| Sesgo | Evidencia | Medida adoptada |
|---|---|---|
| Representación | 7 de 28 barrios con menos de 30 casas; `Landmrk` tiene una | Revisión manual en barrios sub-representados |
| Histórico o geográfico | Razón de 3,6 a 1 en precio entre barrios; 2,4 a 1 controlando por superficie | Equidad territorial como KPI del criterio de salida |
| Proxy geográfico | Las variables físicas permiten reconstruir el barrio (V de Cramér 0,51) | Conservar `Neighborhood` para poder auditar |
| Medición | `Overall Qual` es una nota subjetiva; solo el 64,8 % se explica por variables físicas | Documentar |
| Temporal | Datos de 2006 a 2010 | El modelo tasa a precios de ese período |
| Selección | 18 % de las ventas no son de mercado libre | Alcance limitado al valor de mercado en venta normal |
| Asimetría del error | Brecha de 16,9 puntos entre barrios con el precio en dólares y 8,4 en escala log | Entrenar sobre `log1p(SalePrice)` |

Quitar `Neighborhood` no elimina el sesgo geográfico, porque las demás variables reconstruyen el barrio, y además impide auditarlo.

### 14.2 Lo que mostró esta entrega

- El modelo elegido cumple el umbral de equidad: 6,1 puntos de diferencia de MAPE entre el barrio mejor y el peor predicho en el test real y 6,2 con venta normal, frente a un máximo de 8.
- Random Forest no cumple la equidad en ninguna de las dos entregas, y su peor barrio es `IDOTRR`, el de más zonas no residenciales y uno de los de más ventas anómalas.
- La equidad de los ensambles con venta normal es frágil: cumple o no según el sistema operativo en que se entrena XGBoost (sección 15.3). Es otra razón para no reemplazar a Lasso.
- Quedan dos sesgos medidos en el modelo final: sobreestima en 3,4 % las casas del cuartil más barato y subestima 8 de 9 casas sobre 450.000 USD. La herramienta de tasación los marca con alertas.

### 14.3 Ética y privacidad

- El dataset no tiene nombres ni direcciones. `PID` permitía llegar a la dirección y al propietario a través de registros públicos y se eliminó.
- Quien es tasado por un algoritmo debería poder saber qué factores determinaron el valor. Por eso la regla de elección favorece al modelo interpretable cuando la diferencia de precisión es pequeña, y `tasar()` entrega el desglose de cada estimación.
- El modelo apoya la tasación, no la reemplaza. No debe usarse para decidir solo un crédito hipotecario.
- Los dos errores no cuestan lo mismo: subvalorar perjudica al vendedor y sobrevalorar al comprador y al banco. La sobreestimación del segmento más barato afecta a los compradores con menos recursos, y por eso se reporta como limitación y genera una alerta.

## 15. Problemas que tuvimos durante el proyecto

Esta sección junta lo que no salió como esperábamos y qué hicimos en cada caso.

### 15.1 Reproducir la EP2 antes de empezar

Antes de buscar hiperparámetros, volvimos a ejecutar el notebook de la EP2 para confirmar que partíamos del mismo punto. Con las versiones de las librerías de la ejecución en Colab de la EP2 (numpy 2.1.3, pandas 2.2.3, scikit-learn 1.6.1 y xgboost 3.4.1), los resultados se reprodujeron al dólar. Con las versiones más nuevas de cada librería, Random Forest dejaba de coincidir con los resultados guardados y el notebook se detenía en un control.

Qué hicimos: fijamos las versiones exactas en `requirements.txt` y agregamos al inicio del notebook una celda que compara las versiones instaladas con las requeridas y se detiene si no coinciden.

### 15.2 Colab recalculaba todo sin avisar

Al ejecutar la parte 3 en Colab, la celda del diagnóstico de Random Forest tardaba demasiado. La celda de entorno clonaba el repositorio de la EP2, que no tenía la carpeta `models/busquedas/`, y el notebook repetía todas las búsquedas en silencio. Con los 2 núcleos de Colab eso podía tomar horas.

Qué hicimos: primero el notebook pasó a detenerse con un aviso si faltaban los resultados. En la versión final, la celda de entorno ya no clona ningún repositorio: en Colab pide `requirements.txt` y `ames_clean.csv`, recalcula lo que falte y al final comprime `models/` e `images/` para descargarlos. No medimos cuánto tarda la ejecución completa en Colab.

### 15.3 Windows y Linux dan resultados distintos en XGBoost

Fue el problema más grande. La parte 3 se ejecutó primero en Linux y después, al integrar la parte 4, el notebook completo se ejecutó en Windows, con las mismas versiones de todas las librerías. Las búsquedas de XGBoost dieron otros números y otra configuración ganadora:

| | Linux | Windows |
|---|---|---|
| XGBoost con la configuración de la EP2 (sin submuestreo) | 22.351 USD | 22.351 USD |
| Configuración ganadora de XGBoost | profundidad 3, 1.500 árboles, `colsample_bytree` 0,3 | profundidad 2, 2.000 árboles, `colsample_bytree` 0,5 |
| RMSE de validación de la ganadora | 20.011 USD | 19.809 USD |
| La ganadora de Linux evaluada en Windows | | 19.998 USD |
| La ganadora de Windows evaluada en Linux | 20.060 USD | |
| Random Forest con la configuración de la EP2 | 25.149 USD | 25.151 USD |
| Ridge y Lasso | iguales | iguales |

La diferencia aparece solo cuando XGBoost submuestrea filas o columnas: sin submuestreo los dos sistemas dan lo mismo al dólar. Random Forest difiere en pocos dólares y los lineales no cambian. No investigamos la causa exacta dentro de XGBoost.

Como XGBoost tenía cientos de configuraciones casi empatadas (sección 10.6), una diferencia de pocos cientos de dólares bastó para cambiar cuál quedaba primera. Después ejecutamos en Linux el notebook final, con las búsquedas guardadas en Windows, para ver cuánto cambiaba la parte 4:

| Test | Windows | Linux |
|---|---|---|
| RMSE de XGBoost, test real | 18.241 USD | 17.695 USD |
| RMSE del stacking, test real | 17.903 USD | 17.574 USD |
| RMSE de Lasso (`alpha` de la EP2), test real | 19.241 USD | 19.241 USD |
| Brecha entre barrios de la votación, venta normal | 7,4 (cumple) | 8,5 (no cumple) |
| Brecha entre barrios del stacking, venta normal | 7,5 (cumple) | 8,6 (no cumple) |
| Modelo elegido | Lasso (`alpha` de la EP2) | Lasso (`alpha` de la EP2) |

Qué hicimos: dejamos como oficial la ejecución de Windows, que es la del notebook integrado y con la que está escrito el texto, y documentamos la diferencia en el notebook (sección 32 y advertencia 6 de la sección 52.4) y aquí. La conclusión no cambia en ninguno de los dos sistemas, porque el modelo elegido es lineal y da lo mismo en ambos. Lo que sí cambia es la lectura de los ensambles: su cumplimiento del KPI de equidad con venta normal depende del sistema, y por eso lo tratamos como frágil.

### 15.4 La grilla ampliada de XGBoost quedó pensada para la otra ejecución

La grilla ampliada de XGBoost se definió en el código mirando los bordes de la ganadora de Linux: profundidad 3 y 4, `subsample` de 0,7 a 1,0 y `colsample_bytree` de 0,1 a 0,3. En Windows la ganadora quedó en otros bordes (profundidad 2 y 2.000 árboles), así que esa grilla no los cubre. Lo revisamos con las curvas de validación de la sección 33: con profundidad 1 el error sube 873 USD y con 3.000 árboles baja 58 USD, que es ruido de los folds. La búsqueda no dejó fuera una zona mejor, pero en una búsqueda nueva la grilla ampliada se tendría que definir con los bordes de la ejecución actual.

### 15.5 Codificación de archivos en Windows

El código de la parte 3 lee y escribe los JSON sin indicar la codificación. En Windows Python usa cp1252 por defecto, así que `models/mejores_params.json` quedó con los acentos en cp1252 y no en UTF-8. El notebook funciona porque escribe y lee ese archivo en el mismo sistema, pero un programa que lo lea como UTF-8 falla. La solución es indicar `encoding='utf-8'` en esas lecturas y escrituras, o ejecutar con `PYTHONUTF8=1`, como dice la advertencia 4 de la sección 52.4.

### 15.6 Afinar con las curvas era perseguir ruido

Al principio pensamos usar las curvas de validación para seguir ajustando: si un valor vecino mejoraba en 4 de 5 folds, se adoptaba. Al aplicarlo, los cambios que aparecían eran de menos de 100 USD, muy por debajo de la variación entre folds, y cada cambio abría otra curva con otro vecino un poco mejor. Lo cambiamos por el criterio actual: un vecino solo cuenta si mejora en al menos 4 folds y por más que la desviación entre folds, y en ese caso se amplía la búsqueda en vez de cambiar el valor a mano. En la ejecución final ningún vecino lo cumple.

### 15.7 Dos `alpha` de Lasso equivalentes

La búsqueda de la EP3 eligió para Lasso un `alpha` de 0,000127, con 3 USD menos de error que el 0,000174 de la EP2 pero 111 columnas en vez de 105. Una diferencia de 3 USD es ruido y la regla de interpretabilidad prefiere menos columnas, así que antes de abrir el test agregamos el `alpha` de la EP2 como candidato explícito. Por eso el modelo final tiene el mismo `alpha` que la EP2.

## 16. Conclusiones, limitaciones y recomendaciones

### 16.1 Conclusiones

El modelo final es Lasso con `alpha` = 0,000174 y 105 columnas en uso. Cumple los seis KPI en los dos escenarios de test:

| KPI | Umbral | Test real | Venta normal impuesta |
|---|---|---|---|
| R² | ≥ 0,85 | 0,948 | 0,949 |
| RMSE | ≤ 30.000 USD | 19.241 | 19.084 |
| MAE | ≤ 20.000 USD | 12.407 | 12.462 |
| MAPE | ≤ 12 % | 6,77 % | 6,93 % |
| Brecha de R² train-test | ≤ 5 puntos | −0,9 | −1,0 |
| Brecha de MAPE entre barrios | ≤ 8 puntos | 6,1 | 6,2 |

| Parte | Qué se hizo | Resultado |
|---|---|---|
| EP2 | Preparación por familia de modelos y comparación con parámetros iniciales | Lasso elegido; los árboles sobreajustaban |
| EP3, hiperparámetros | Búsqueda aleatoria, grillas y curvas de validación en los cuatro modelos | XGBoost deja de sobreajustar y baja 2.542 USD; queda casi empatado con Lasso. Random Forest cumple la brecha pero no mejora. Los lineales no cambian |
| EP3, ensambles | Bagging, boosting, votación y stacking | Los ensambles vuelven competitivos a los árboles y el stacking supera a Lasso por poco (0,66 puntos de R²) |
| EP3, generalización | Folds nuevos, validación anidada, curvas de aprendizaje y test único | Ningún modelo que cumple la brecha rinde peor con casas nuevas; XGBoost tiene un optimismo moderado (+440 USD) |
| EP3, selección | Regla de la EP2 aplicada en código | Se mantiene Lasso por la regla, la trazabilidad y la robustez |

La búsqueda de hiperparámetros no cambió el modelo final, pero sí cambió la razón para elegirlo. En la EP2, XGBoost quedaba fuera porque sobreajustaba y tenía más error. Ahora cumple todos los KPI en el test real y tiene menos error que Lasso, y queda fuera porque la ventaja es menor que la variación entre folds y no compensa perder el desglose de cada tasación.

### 16.2 Limitaciones

- Precios de 2006 a 2010, incluida la crisis de 2008: el modelo tasa a precios de ese período.
- Subestima las casas de más de 450.000 USD (8 de 9 en test, −12,5 % en promedio) y sobreestima el cuartil más barato (+3,4 %). Ningún ensamble lo corrigió.
- Ventas anómalas (`Abnorml`, `Family`): error de 11 % a 16 %, y no se pueden identificar antes de vender.
- Las zonas no residenciales quedan fuera de alcance.
- El rango orientativo no es un intervalo de confianza formal y es más estrecho de lo real en los extremos.
- La codificación de las variables no entró a la búsqueda de hiperparámetros.
- No comparamos los ensambles con un árbol de decisión ajustado por separado (sección 11.1).
- Los resultados de XGBoost y de los ensambles que lo usan dependen del sistema operativo (sección 15.3).

### 16.3 Recomendaciones para el negocio

1. Usar el modelo como valor de partida del perito en viviendas de zonas residenciales, con la venta normal impuesta y `Yr Sold = 2010`, junto con el rango y el desglose de `tasar()`.
2. Revisión manual obligatoria en los casos con alerta: valor sobre 450.000 USD, cuartil barato, datos atípicos y zonas no residenciales.
3. Antes de usarlo con precios actuales, reentrenar con datos recientes y con train y test juntos (2.896 casas residenciales).
4. Monitorear el error por barrio, en especial `IDOTRR`, `Edwards`, `BrkSide` y `ClearCr`, y reentrenar de forma periódica con las ventas nuevas.
5. Si en el futuro se prioriza la precisión sobre la trazabilidad, el candidato es el stacking de Lasso y XGBoost (1.338 USD menos de RMSE en test), con SHAP para explicar cada tasación, siempre que mantenga la equidad entre barrios con venta normal y que se fije el sistema en que se entrena.

## 17. Reproducibilidad

Estructura del proyecto:

```
├── data/
│   ├── raw/Ames_Iowa_Housing_Dataset.csv     datos originales
│   └── processed/                            ames_clean.csv, conjuntos por modelo y segmentos.csv
├── docs/
│   └── informe_tecnico.md                    este informe
├── images/                                   figuras de las partes 3 (p3_*) y 4 (p4_*)
├── models/
│   ├── busquedas/                            resultados de cada búsqueda de la parte 3 (36 JSON)
│   ├── parte4/                               validaciones, curvas, predicciones fuera de fold y resultados del test
│   ├── mejores_params.json                   hiperparámetros elegidos en la parte 3
│   ├── modelo_final.joblib                   modelo final entrenado con todo train
│   └── modelo_final.json                     ficha del modelo final
├── notebooks/
│   └── Housing_Prices_ML_P3_v5.ipynb         notebook completo (partes 1 a 4)
├── requirements.txt
└── README.md
```

Cómo ejecutarlo está en el `README.md`. Puntos que afectan la reproducibilidad:

- Las versiones de numpy, pandas, scikit-learn y xgboost tienen que ser las de `requirements.txt` (Python 3.12 o 3.13). La celda de entorno se detiene si no coinciden.
- Con las carpetas `models/busquedas/` y `models/parte4/` presentes, el notebook carga los resultados guardados. En Linux, con 16 núcleos, la ejecución completa tomó 6 minutos. Sin esas carpetas recalcula todo; repetir solo las búsquedas de la parte 3 tomó 18 minutos con 12 núcleos.
- Todas las cifras del notebook y de este informe (salvo la sección 15.3) salen de la ejecución en Windows. En Linux o en Colab, Ridge y Lasso dan lo mismo, Random Forest cambia en pocos dólares y XGBoost y los ensambles cambian en cientos de dólares.
- Las celdas de control de la parte 3 imprimen la diferencia entre lo calculado y lo guardado, pero no detienen la ejecución. En Linux, el control de XGBoost muestra +251 USD.

## 18. Referencias

- De Cock, D. (2011). Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project. Journal of Statistics Education, 19(3).
- Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.
- Friedman, J. H. (2001). Greedy Function Approximation: A Gradient Boosting Machine. The Annals of Statistics, 29(5), 1189-1232.
- Friedman, J. H. (2002). Stochastic Gradient Boosting. Computational Statistics & Data Analysis, 38(4), 367-378.
- Chen, T. y Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785-794.
- Bergstra, J. y Bengio, Y. (2012). Random Search for Hyper-Parameter Optimization. Journal of Machine Learning Research, 13, 281-305.
- Chapman, P. et al. (2000). CRISP-DM 1.0: Step-by-step data mining guide.
- Informes de la EP1 y la EP2 del proyecto.
