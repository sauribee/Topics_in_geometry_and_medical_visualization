# Informe: B-Splines Cuadráticos para Contornos en Imágenes Médicas

## Introducción

Este informe detalla la implementación de B-splines cuadráticos para la interpolación de contornos en imágenes médicas, específicamente para resonancias magnéticas del manguito rotador del hombro. La necesidad de representar estos contornos anatómicos de manera precisa es crucial para el diagnóstico y análisis de posibles desgarros musculares, una patología común en Colombia.

## Contexto Médico

El manguito rotador es un conjunto de músculos y tendones que rodean la articulación del hombro, proporcionando estabilidad y permitiendo un amplio rango de movimiento. Las lesiones en esta área son frecuentes, especialmente en personas que realizan movimientos repetitivos con los brazos.

Las resonancias magnéticas del hombro proporcionan cortes axiales (15-20 típicamente) que permiten visualizar el estado de estos tejidos. La delimitación precisa del contorno del manguito rotador en estas imágenes es fundamental para:

1. Cuantificar el grado de lesión
2. Planificar tratamientos
3. Monitorear la evolución post-tratamiento
4. Realizar análisis comparativos

## Fundamentos Teóricos

### B-Splines Cuadráticos

Los B-splines son funciones paramétricas definidas a trozos que proporcionan un alto grado de continuidad. Un B-spline cuadrático (grado 2) garantiza continuidad C¹ en toda la curva, lo que significa que tanto la curva como su primera derivada son continuas.

La fórmula general de un B-spline es:

$$S(t) = \sum_{i=0}^{n} P_i N_{i,k}(t)$$

Donde:
- $P_i$ son los puntos de control
- $N_{i,k}(t)$ son las funciones base de B-spline de grado $k$
- $t$ es el parámetro que recorre la curva

Para un B-spline cuadrático, las funciones base se definen recursivamente, proporcionando una interpolación suave entre los puntos de control.

### Parametrización por Longitud de Cuerda

Un aspecto crucial en la interpolación de contornos anatómicos es la distribución no uniforme de los puntos. Para abordar este problema, implementamos una parametrización por longitud de cuerda, que asigna valores de parámetro proporcionales a la distancia entre puntos consecutivos.

La parametrización se calcula como:

$$t_i = \frac{\sum_{j=1}^{i} d_j}{\sum_{j=1}^{n} d_j}$$

Donde:
- $d_j$ es la distancia euclidiana entre los puntos $j-1$ y $j$
- $t_i$ es el valor del parámetro asignado al punto $i$

Esta parametrización asegura que la curva se adapte mejor a la distribución espacial de los puntos, asignando más "tiempo" paramétrico a las regiones donde los puntos están más separados.

## Implementación

### Estructura del Código

La implementación consta de los siguientes componentes principales:

1. **QuadraticBSpline**: Clase que implementa la interpolación con B-splines cuadráticos.
2. **Funciones de carga de datos**: Para importar los puntos desde archivos MATLAB.
3. **Funciones de visualización**: Para representar gráficamente los resultados.

### Algoritmo de Interpolación

El proceso de interpolación sigue estos pasos:

1. **Carga de puntos**: Se importan los puntos del contorno desde un archivo MATLAB.
2. **Cálculo de la parametrización**: Se implementa la parametrización por longitud de cuerda.
3. **Construcción del B-spline**: Se utiliza `scipy.interpolate.splprep` para construir el B-spline.
4. **Evaluación de la curva**: Se evalúa el B-spline en un conjunto denso de valores del parámetro.
5. **Visualización**: Se representa gráficamente la curva junto con los puntos originales.

### Manejo de Curvas Cerradas

Un aspecto importante es el manejo de curvas cerradas, ya que los contornos anatómicos suelen ser estructuras cerradas. Para ello:

1. Se añaden puntos adicionales al principio y al final para garantizar la continuidad.
2. Se utilizan condiciones de periodicidad en el ajuste del B-spline.
3. Se asegura que la curva se cierre suavemente, sin discontinuidades.

### Cálculo y Visualización de Derivadas

La implementación permite calcular y visualizar las derivadas (tangentes) a lo largo de la curva, lo que proporciona información adicional sobre la forma y suavidad del contorno. Las derivadas se normalizan para su visualización como vectores unitarios.

## Resultados

### Visualización del Contorno

El resultado principal es una curva B-spline que interpola con precisión los puntos del contorno del manguito rotador, proporcionando una representación suave y continua. La curva se ajusta a la distribución no uniforme de los puntos gracias a la parametrización por longitud de cuerda.

### Análisis de la Parametrización

La parametrización por longitud de cuerda distribuye los valores del parámetro de manera proporcional a las distancias entre puntos consecutivos. Esto se refleja en los valores calculados:

```
Parameter values: [0.0, 0.043, 0.096, 0.150, ..., 0.967, 1.0]
Chord lengths: [18.38, 22.20, 22.83, 20.10, ..., 11.70, 14.04]
```

Esta distribución no uniforme del parámetro permite que la curva se adapte mejor a regiones con diferentes densidades de puntos.

## Ventajas de los B-Splines para Contornos Médicos

1. **Representación suave**: Los B-splines proporcionan curvas suaves que representan mejor las estructuras anatómicas naturales.
2. **Continuidad garantizada**: La continuidad C¹ asegura transiciones suaves, eliminando artefactos que podrían interpretarse erróneamente como anomalías.
3. **Eficiencia computacional**: La representación paramétrica es computacionalmente eficiente para operaciones posteriores como cálculo de áreas o análisis de curvatura.
4. **Adaptabilidad**: La parametrización por longitud de cuerda permite adaptar la curva a la distribución espacial de los puntos.
5. **Robustez**: El algoritmo implementado incluye mecanismos de manejo de errores y casos especiales, como puntos muy cercanos o configuraciones problemáticas.

## Limitaciones y Trabajo Futuro

### Limitaciones actuales

1. **Dependencia del número de puntos**: La calidad de la interpolación depende del número y distribución de los puntos originales.
2. **Ajuste local vs. global**: Los B-splines proporcionan un ajuste que puede no ser óptimo desde una perspectiva global.

### Trabajo futuro

1. **Optimización de puntos**: Similar al trabajo realizado en el Assignment 2, se podría implementar un algoritmo para optimizar la posición de los puntos de control.
2. **Splines de mayor grado**: Implementar B-splines de grado superior para aplicaciones que requieran mayor suavidad.
3. **Análisis de curvatura**: Desarrollar herramientas para analizar la curvatura a lo largo del contorno, lo que podría proporcionar información diagnóstica adicional.
4. **Segmentación automática**: Integrar esta técnica con algoritmos de segmentación automática de imágenes médicas.

## Conclusiones

La implementación de B-splines cuadráticos con parametrización por longitud de cuerda proporciona una solución eficaz para la representación de contornos en imágenes médicas. La continuidad y suavidad garantizadas por los B-splines, junto con la adaptabilidad proporcionada por la parametrización no uniforme, hacen que esta técnica sea particularmente adecuada para aplicaciones médicas donde la precisión y la naturalidad son cruciales.

Este enfoque sienta las bases para desarrollos futuros en análisis de imágenes médicas, proporcionando una representación matemáticamente robusta que puede utilizarse para cuantificación, comparación y visualización avanzada de estructuras anatómicas.

## Referencias

1. de Boor, C. (1978). A Practical Guide to Splines. Springer-Verlag.
2. Piegl, L., & Tiller, W. (1997). The NURBS Book. Springer-Verlag.
3. Bartels, R. H., Beatty, J. C., & Barsky, B. A. (1987). An Introduction to Splines for Use in Computer Graphics and Geometric Modeling. Morgan Kaufmann.
4. Lee, E. T. Y. (1989). Choosing nodes in parametric curve interpolation. Computer-Aided Design, 21(6), 363-370. 