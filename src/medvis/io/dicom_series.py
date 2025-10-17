import SimpleITK as sitk
from typing import Sequence


def read_series_to_volume(dicom_dir: str, *, load_private: bool = False) -> sitk.Image:
    reader = sitk.ImageSeriesReader()
    series = reader.GetGDCMSeriesIDs(dicom_dir)
    if not series:
        raise FileNotFoundError(f"No hay series en {dicom_dir}")
    files: Sequence[str] = reader.GetGDCMSeriesFileNames(dicom_dir, series[0])
    reader.SetFileNames(files)
    if load_private:
        reader.MetaDataDictionaryArrayUpdateOn()
        reader.LoadPrivateTagsOn()
    return reader.Execute()
