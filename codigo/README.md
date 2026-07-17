# Codigo del taller

Esta carpeta organiza el pipeline por partes del taller. La logica fue separada
desde el script monolitico original de forma conservadora para mantener el
mismo orden de ejecucion y las mismas salidas.

Ejecutar todo:

```terminal
& .\.env\Scripts\python.exe codigo\99_ejecutar_todo.py
```

Ejecutar una parte individual:

```terminal
& .\.env\Scripts\python.exe codigo\03_parte_iii_regresion_logistica.py
```

Si una parte depende de resultados previos, carga archivos intermedios desde
`resultados/intermedios/` sin recalcular las secciones anteriores. Por ejemplo,
`03_parte_iii_regresion_logistica.py` usa `parte_ii.pkl` y arranca directamente
en la Pregunta 11.

Para regenerar todo desde cero y actualizar esos intermedios, use siempre:

```terminal
& .\.env\Scripts\python.exe codigo\99_ejecutar_todo.py
```

Los resultados se escriben en `resultados/` para mantener una unica ubicacion
de salidas del proyecto.

Orden de los scripts:

1. `00_configuracion.py`
2. `01_parte_i_clasificacion_multiple.py`
3. `02_parte_ii_clasificacion_binaria.py`
4. `03_parte_iii_regresion_logistica.py`
5. `04_parte_iv_analisis_espacial.py`
6. `05_parte_v_propuesta_libre.py`
