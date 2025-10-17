from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version(
        "medvis"
    )  # debe coincidir con [project].name en pyproject.toml
except PackageNotFoundError:
    # fallback para ejecución directa sin instalar
    __version__ = "0.1.0"
