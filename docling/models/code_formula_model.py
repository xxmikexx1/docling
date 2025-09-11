import re
from collections.abc import Iterable
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
from docling_core.types.doc import (
    CodeItem,
    DocItemLabel,
    DoclingDocument,
    NodeItem,
    TextItem,
)
from docling_core.types.doc.labels import CodeLanguageLabel
from PIL import Image
from pydantic import BaseModel
from transformers import AutoModelForImageTextToText, AutoProcessor

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import ItemAndImageEnrichmentElement
from docling.models.base_model import BaseItemAndImageEnrichmentModel
from docling.models.utils.hf_model_download import download_hf_model
from docling.utils.accelerator_utils import decide_device


class CodeFormulaModelOptions(BaseModel):
    """Configuration options for the `CodeFormulaModel`.

    Attributes:
        kind: The type of the model, fixed to "code_formula".
        do_code_enrichment: If `True`, enables the enrichment of code blocks.
        do_formula_enrichment: If `True`, enables the enrichment of formulas.
    """

    kind: Literal["code_formula"] = "code_formula"
    do_code_enrichment: bool = True
    do_formula_enrichment: bool = True


class CodeFormulaModel(BaseItemAndImageEnrichmentModel):
    """A model for recognizing and transcribing code and mathematical formulas from images.

    This model uses a vision-encoder-decoder architecture to process images of
    code blocks and formulas, converting them into structured text (e.g., LaTeX
    for formulas).

    Attributes:
        enabled: A boolean indicating if the model is enabled.
        options: A `CodeFormulaModelOptions` object for configuration.
        device: The accelerator device (e.g., "cuda", "cpu") to run the model on.
    """

    _model_repo_folder = "ds4sd--CodeFormulaV2"
    elements_batch_size = 5
    images_scale = 1.67  # = 120 dpi, aligned with training data resolution
    expansion_factor = 0.18

    def __init__(
        self,
        enabled: bool,
        artifacts_path: Optional[Path],
        options: CodeFormulaModelOptions,
        accelerator_options: AcceleratorOptions,
    ):
        """Initializes the CodeFormulaModel.

        Args:
            enabled: A boolean flag to enable or disable the model.
            artifacts_path: An optional path to the directory containing the
                model artifacts. If not provided, the model will be downloaded.
            options: The configuration options for the model.
            accelerator_options: The hardware acceleration options.
        """
        self.enabled = enabled
        self.options = options

        if self.enabled:
            self.device = decide_device(
                accelerator_options.device,
                supported_devices=[AcceleratorDevice.CPU, AcceleratorDevice.CUDA],
            )

            if artifacts_path is None:
                artifacts_path = self.download_models()
            else:
                artifacts_path = artifacts_path / self._model_repo_folder

            self._processor = AutoProcessor.from_pretrained(
                artifacts_path,
            )
            self._model_max_length = self._processor.tokenizer.model_max_length
            self._model = AutoModelForImageTextToText.from_pretrained(
                artifacts_path, device_map=self.device
            )
            self._model.eval()

    @staticmethod
    def download_models(
        local_dir: Optional[Path] = None,
        force: bool = False,
        progress: bool = False,
    ) -> Path:
        """Downloads the CodeFormulaV2 model from Hugging Face.

        Args:
            local_dir: An optional local directory to save the model to.
            force: If `True`, forces the re-download of the model.
            progress: If `True`, displays a progress bar.

        Returns:
            The path to the local directory where the model is saved.
        """
        return download_hf_model(
            repo_id="ds4sd/CodeFormulaV2",
            revision="main",
            local_dir=local_dir,
            force=force,
            progress=progress,
        )

    def is_processable(self, doc: DoclingDocument, element: NodeItem) -> bool:
        """Determines if a given element can be processed by this model.

        This model can process `CodeItem` elements (if code enrichment is enabled)
        and `TextItem` elements with the label `FORMULA` (if formula enrichment
        is enabled).

        Args:
            doc: The `DoclingDocument` being processed.
            element: The `NodeItem` to check.

        Returns:
            `True` if the element is processable, `False` otherwise.
        """
        return self.enabled and (
            (isinstance(element, CodeItem) and self.options.do_code_enrichment)
            or (
                isinstance(element, TextItem)
                and element.label == DocItemLabel.FORMULA
                and self.options.do_formula_enrichment
            )
        )

    def _extract_code_language(self, input_string: str) -> Tuple[str, Optional[str]]:
        """Extracts a programming language from the beginning of a string.

        This function checks if the input string starts with a pattern of the form
        ``<_some_language_>``. If it does, it extracts the language string and returns
        a tuple of (remainder, language). Otherwise, it returns the original string
        and `None`.

        Args:
            input_string (str): The input string, which may start with ``<_language_>``.

        Returns:
            Tuple[str, Optional[str]]:
                A tuple where:
                - The first element is either:
                    - The remainder of the string (everything after ``<_language_>``),
                    if a match is found; or
                    - The original string, if no match is found.
                - The second element is the extracted language if a match is found;
                otherwise, `None`.
        """
        pattern = r"^<_([^_>]+)_>\s*(.*)"
        match = re.match(pattern, input_string, flags=re.DOTALL)
        if match:
            language = str(match.group(1))  # the captured programming language
            remainder = str(match.group(2))  # everything after the <_language_>
            return remainder, language
        else:
            return input_string, None

    def _get_code_language_enum(self, value: Optional[str]) -> CodeLanguageLabel:
        """
        Converts a string to a corresponding `CodeLanguageLabel` enum member.

        If the provided string does not match any value in `CodeLanguageLabel`,
        it defaults to `CodeLanguageLabel.UNKNOWN`.

        Args:
            value (Optional[str]): The string representation of the code language or None.

        Returns:
            CodeLanguageLabel: The corresponding enum member if the value is valid,
            otherwise `CodeLanguageLabel.UNKNOWN`.
        """
        if not isinstance(value, str):
            return CodeLanguageLabel.UNKNOWN

        try:
            return CodeLanguageLabel(value)
        except ValueError:
            return CodeLanguageLabel.UNKNOWN

    def _get_prompt(self, label: str) -> str:
        """
        Constructs the prompt for the model based on the input label.

        Parameters
        ----------
        label : str
            The type of input, either 'code' or 'formula'.

        Returns
        -------
        str
            The constructed prompt including necessary tokens and query.

        Raises
        ------
        NotImplementedError
            If the label is not 'code' or 'formula'.
        """
        if label == "code":
            query = "<code>"
        elif label == "formula":
            query = "<formula>"
        else:
            raise NotImplementedError("Label must be either code or formula")

        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": query}],
            },
        ]

        prompt = self._processor.apply_chat_template(
            messages, add_generation_prompt=True
        )

        return prompt

    def _post_process(self, texts: list[str]) -> list[str]:
        """
        Processes a list of text strings by truncating at '<end_of_utterance>' and
        removing a predefined set of unwanted substrings.

        Parameters
        ----------
        texts : list[str]
            A list of strings to be post-processed.

        Returns
        -------
        list[str]
            A list of cleaned strings with specified substrings removed and truncated at
                '<end_of_utterance>' if present.
        """
        to_remove = ["</code>", "</formula>", "<loc_0><loc_0><loc_500><loc_500>"]

        def clean_text(text: str) -> str:
            idx = text.find("<end_of_utterance>")
            if idx != -1:
                text = text[:idx]

            for token in to_remove:
                if token in text:
                    text = text.replace(token, "")
            return text.lstrip()

        return [clean_text(t) for t in texts]

    def __call__(
        self,
        doc: DoclingDocument,
        element_batch: Iterable[ItemAndImageEnrichmentElement],
    ) -> Iterable[NodeItem]:
        """Processes a batch of elements and enriches them with predictions.

        This method takes a batch of `ItemAndImageEnrichmentElement` objects,
        runs them through the CodeFormulaV2 model, and updates the `text` and
        `code_language` attributes of the original items with the model's output.

        Args:
            doc: The `DoclingDocument` being processed.
            element_batch: An iterable of elements to be enriched.

        Returns:
            An iterable of the enriched `NodeItem`s.
        """
        if not self.enabled:
            for element in element_batch:
                yield element.item
            return

        labels: List[str] = []
        images: List[Union[Image.Image, np.ndarray]] = []
        elements: List[TextItem] = []
        for el in element_batch:
            elements.append(el.item)  # type: ignore[arg-type]
            labels.append(el.item.label)  # type: ignore[attr-defined]
            images.append(el.image)

        prompts = [self._get_prompt(label) for label in labels]
        inputs = self._processor(
            text=prompts,
            images=images,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        gen_kwargs = dict(
            max_new_tokens=self._model_max_length - inputs.input_ids.shape[1],
            use_cache=True,
            do_sample=False,
        )

        generated_ids = self._model.generate(**inputs, **gen_kwargs)

        outputs = self._processor.batch_decode(
            generated_ids[:, inputs.input_ids.shape[1] :], skip_special_tokens=False
        )
        outputs = self._post_process(outputs)

        for item, output in zip(elements, outputs):
            if isinstance(item, CodeItem):
                output, code_language = self._extract_code_language(output)
                item.code_language = self._get_code_language_enum(code_language)
            item.text = output

            yield item
