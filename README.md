# Redes Neuronales Artificiales

Taller de perceptron multicapa para la clasificacion de severidad de
*Verticillium* sp. mediante indices espectrales.

## Instalacion

Desde la terminal, ubicado en la raiz del proyecto, cree y active el entorno
virtual:

```terminal
python -m venv .env
& .\.env\Scripts\Activate.ps1
```

Si la terminal bloquea la activacion del entorno, habilite la ejecucion solo
para la sesion actual y vuelva a activar:

```terminal
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
& .\.env\Scripts\Activate.ps1
```

Luego instale las dependencias del proyecto:

```terminal
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Para verificar que las dependencias quedaron instaladas correctamente:

```terminal
python -m pip check
```

## Orden de ejecucion

El flujo recomendado es ejecutar todo el proyecto desde el script maestro:

```terminal
& .\.env\Scripts\python.exe codigo\99_ejecutar_todo.py
```

Ese comando ejecuta los scripts en este orden:

1. `codigo/00_configuracion.py`: carga librerias, rutas, constantes y funciones comunes.
2. `codigo/01_parte_i_clasificacion_multiple.py`: exploracion, preprocesamiento, particiones, MLP multiclase y metricas.
3. `codigo/02_parte_ii_clasificacion_binaria.py`: colapso binario sano/enfermo, MLP binario, umbral y ROC.
4. `codigo/03_parte_iii_regresion_logistica.py`: regresion logistica, seleccion de variables e importancia por permutacion.
5. `codigo/04_parte_iv_analisis_espacial.py`: grilla, Moran global e incorporacion de coordenadas.
6. `codigo/05_parte_v_propuesta_libre.py`: propuesta libre, dropout y Random Forest.

## Ejecucion por partes

Despues de ejecutar `99_ejecutar_todo.py` al menos una vez, tambien se puede
trabajar una seccion individual:

```terminal
& .\.env\Scripts\python.exe codigo\03_parte_iii_regresion_logistica.py
```

Los scripts individuales cargan sus dependencias desde `resultados/intermedios/`
sin recalcular las partes anteriores. Si falta un intermedio, el script indicara
que parte debe ejecutarse primero.

## Salidas

Las tablas, figuras e intermedios se guardan en:

- `resultados/tablas/`
- `resultados/figuras/`
- `resultados/intermedios/`
