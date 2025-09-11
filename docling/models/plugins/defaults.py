def ocr_engines():
    """Returns a dictionary of the default OCR engine classes.

    This function is used as a plugin entry point to register the built-in OCR
    models with the `OcrFactory`.

    Returns:
        A dictionary where the key "ocr_engines" maps to a list of OCR model
        classes.
    """
    from docling.models.easyocr_model import EasyOcrModel
    from docling.models.ocr_mac_model import OcrMacModel
    from docling.models.rapid_ocr_model import RapidOcrModel
    from docling.models.tesseract_ocr_cli_model import TesseractOcrCliModel
    from docling.models.tesseract_ocr_model import TesseractOcrModel

    return {
        "ocr_engines": [
            EasyOcrModel,
            OcrMacModel,
            RapidOcrModel,
            TesseractOcrModel,
            TesseractOcrCliModel,
        ]
    }


def picture_description():
    """Returns a dictionary of the default picture description model classes.

    This function is used as a plugin entry point to register the built-in
    picture description models with the `PictureDescriptionFactory`.

    Returns:
        A dictionary where the key "picture_description" maps to a list of
        picture description model classes.
    """
    from docling.models.picture_description_api_model import PictureDescriptionApiModel
    from docling.models.picture_description_vlm_model import PictureDescriptionVlmModel

    return {
        "picture_description": [
            PictureDescriptionVlmModel,
            PictureDescriptionApiModel,
        ]
    }
