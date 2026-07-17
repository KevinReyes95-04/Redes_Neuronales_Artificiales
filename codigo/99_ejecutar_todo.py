"""Ejecuta todas las partes del taller en el mismo orden que la V2 original.

Los archivos por parte comparten un mismo diccionario de variables para conservar
las dependencias entre preguntas y reproducir las salidas actuales.
"""

from pathlib import Path
from _bootstrap import configure_console

configure_console()

SCRIPTS = [
    "00_configuracion.py",
    "01_parte_i_clasificacion_multiple.py",
    "02_parte_ii_clasificacion_binaria.py",
    "03_parte_iii_regresion_logistica.py",
    "04_parte_iv_analisis_espacial.py",
    "05_parte_v_propuesta_libre.py",
]


def run_script(path, state):
    state["__file__"] = str(path)
    state["__name__"] = "__main__"
    code = path.read_text(encoding="utf-8")
    exec(compile(code, str(path), "exec"), state)


def main():
    base_dir = Path(__file__).resolve().parent
    state = {}
    for script in SCRIPTS:
        path = base_dir / script
        print("\n" + "#" * 80)
        print(f"Ejecutando {path.name}")
        print("#" * 80)
        run_script(path, state)


if __name__ == "__main__":
    main()
