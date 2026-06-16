# Pregunta 1: interpretacion del analisis exploratorio

## Descripcion de los datos

El conjunto contiene 212 observaciones y seis categorias ordinales de
severidad, codificadas de 1 a 6. La variable 1 corresponde a plantas sanas y
los valores mayores representan niveles crecientes de severidad.

El archivo incluye cinco predictores: cuatro indices espectrales (`NDVI`,
`EVI`, `NDRE` y `GLI`) y la altura media. Esto difiere del enunciado, que
menciona cinco indices espectrales mas la altura. Ademas, la altura aparece
entre 0.0005 y 0.8472, por lo que probablemente ya se encuentra normalizada y
no expresada directamente en centimetros. Ambas diferencias deben aclararse
con la fuente de los datos.

La tabla completa con media, desviacion estandar, minimo y maximo para cada
predictor y categoria se encuentra en:

`resultados/tablas/04_estadisticos_descriptivos_por_severidad.csv`

Tambien se genero el libro:

`resultados/tablas/analisis_exploratorio_pregunta_1.xlsx`

## Calidad de los datos

No se encontraron valores faltantes ni infinitos. Se identifico una fila
duplicada: las filas originales 174 y 175 contienen exactamente los mismos
valores.

En 48 observaciones, los valores de `EVI`, `GLI` y altura son exactamente
iguales. Esta coincidencia ocurre principalmente en las severidades 3 y 4 y
es poco esperable para tres variables obtenidas de manera independiente. No
se modificaron ni eliminaron estas observaciones, pero deben verificarse
contra los datos de origen antes de entrenar los modelos. Los registros estan
listados en `resultados/tablas/03_registros_para_revision.csv`.

## Diferencias entre categorias

### NDVI

El NDVI muestra el patron ordinal mas claro. Su media disminuye de 0.8784 en
la severidad 1 a 0.6871 en la severidad 6. La asociacion de Spearman con la
severidad es fuerte y negativa (`rho = -0.8548`). Las categorias adyacentes
1 y 2 presentan solapamiento, pero la separacion aumenta a partir de la
categoria 3.

### NDRE

El NDRE tambien disminuye de forma consistente: pasa de una media de 0.3383
en la severidad 1 a 0.1987 en la severidad 6. Su correlacion con la severidad
es `rho = -0.8170`. Junto con el NDVI, parece ser uno de los predictores con
mayor capacidad para distinguir niveles de enfermedad.

### EVI

El EVI disminuye desde 0.7905 en plantas sanas hasta 0.3180 en la severidad
6, con una asociacion negativa moderada a fuerte (`rho = -0.6124`). Sin
embargo, las categorias intermedias tienen alta dispersion y el patron no es
completamente monotono: la media de la severidad 5 es ligeramente mayor que
la de la severidad 4.

### GLI

El GLI presenta una disminucion general, pero con oscilaciones entre las
categorias 3, 4 y 5. Su asociacion con la severidad es debil
(`rho = -0.2535`) y existe un solapamiento considerable. Su contribucion al
modelo podria depender de relaciones no lineales o de interacciones con
otros predictores.

### Altura media

La altura disminuye desde una media de 0.5101 en la severidad 1 hasta 0.2492
en la severidad 4, pero aumenta de nuevo en las categorias 5 y 6. La
correlacion es negativa pero moderada (`rho = -0.3401`) y las distribuciones
se solapan ampliamente. Por si sola, la altura parece tener menor poder
discriminante que NDVI y NDRE.

## Relacion entre predictores

NDVI y NDRE presentan una correlacion de Spearman muy alta (`rho = 0.94`),
lo cual indica que contienen informacion parcialmente redundante. EVI
tambien se relaciona con GLI (`rho = 0.71`) y con la altura (`rho = 0.66`).
Estas asociaciones no impiden utilizar las variables en una red neuronal,
pero deben considerarse al interpretar su importancia.

## Transformacion y estandarizacion

No se recomienda aplicar inicialmente transformaciones logaritmicas. Los
predictores estan acotados aproximadamente entre 0 y 1, y varios contienen
valores muy cercanos a cero. Una transformacion logaritmica dificultaria la
interpretacion y no resolveria el patron de valores que requiere revision.

Si se confirma la validez de los datos, se recomienda estandarizar los cinco
predictores mediante puntuaciones Z antes de entrenar el perceptron
multicapa. Aunque sus rangos son parecidos, sus dispersiones son diferentes;
la estandarizacion facilitara la optimizacion y evitara que una variable
influya mas por su escala numerica.

El escalador debera ajustarse exclusivamente con el conjunto de
entrenamiento y luego aplicarse sin reajuste a validacion y prueba. De esta
manera se evita la fuga de informacion.

## Conclusion

Existen diferencias visibles entre las categorias de severidad. NDVI y NDRE
presentan los patrones mas ordenados y una separacion clara entre plantas
sanas, niveles intermedios y severidades altas. EVI aporta una señal
adicional, aunque con mayor variabilidad. GLI y altura muestran un
solapamiento considerable y relaciones menos monotonicamente ordenadas.

Antes del modelado debe verificarse el origen de las 48 coincidencias entre
EVI, GLI y altura, la fila duplicada y la unidad real de la altura. Estas
observaciones se conservaron en el analisis para describir fielmente el
archivo recibido.
