"""Apoyo comun para ejecutar el taller por partes."""

from pathlib import Path
import pickle
import sys
from types import SimpleNamespace


def configure_console():
    """Evita errores de codificacion al imprimir simbolos en Windows."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def load_config(current_file, namespace):
    """Carga 00_configuracion.py si el script se ejecuta de forma aislada."""
    configure_console()
    if "DATA_PATH" in namespace and "INTERMEDIATE_DIR" in namespace:
        return

    config_path = Path(current_file).resolve().parent / "00_configuracion.py"
    original_file = namespace.get("__file__", str(current_file))
    original_name = namespace.get("__name__", "__main__")

    namespace["__file__"] = str(config_path)
    namespace["__name__"] = "__main__"
    code = config_path.read_text(encoding="utf-8")
    exec(compile(code, str(config_path), "exec"), namespace)

    namespace["__file__"] = original_file
    namespace["__name__"] = original_name


def save_intermediate(namespace, name, keys):
    """Guarda variables necesarias para continuar desde otra parte."""
    path = namespace["INTERMEDIATE_DIR"] / f"{name}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {key: namespace[key] for key in keys if key in namespace}
    with path.open("wb") as file:
        pickle.dump(data, file)


def load_intermediate(namespace, name, required_keys=None, producer=None):
    """Carga un archivo intermedio y valida las variables esperadas."""
    path = namespace["INTERMEDIATE_DIR"] / f"{name}.pkl"
    if not path.exists():
        hint = f" Ejecute primero `{producer}`." if producer else ""
        raise FileNotFoundError(f"No existe el intermedio requerido: {path}.{hint}")

    with path.open("rb") as file:
        data = pickle.load(file)
    namespace.update(data)

    if "h_bin_history" in data:
        namespace["h_bin"] = SimpleNamespace(history=data["h_bin_history"])

    if required_keys:
        missing = [key for key in required_keys if key not in namespace]
        if missing:
            raise KeyError(f"El intermedio {path.name} no contiene: {missing}")


def save_keras_model(namespace, model, filename):
    """Guarda un modelo Keras como artefacto intermedio."""
    path = namespace["INTERMEDIATE_DIR"] / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path, overwrite=True)


def load_keras_model(namespace, variable_name, filename):
    """Carga un modelo Keras intermedio si no existe en memoria."""
    if variable_name in namespace:
        return

    path = namespace["INTERMEDIATE_DIR"] / filename
    if not path.exists():
        raise FileNotFoundError(
            f"No existe el modelo requerido: {path}. "
            "Ejecute primero `python codigo/02_parte_ii_clasificacion_binaria.py`."
        )
    namespace[variable_name] = namespace["keras"].models.load_model(path)
