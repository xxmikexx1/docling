<p align="center">
  <a href="https://github.com/docling-project/docling">
    <img loading="lazy" alt="Docling" src="https://github.com/docling-project/docling/raw/main/docs/assets/docling_processing.png" width="100%"/>
  </a>
</p>

# Docling

<p align="center">
  <a href="https://trendshift.io/repositories/12132" target="_blank"><img src="https://trendshift.io/api/badge/repositories/12132" alt="DS4SD%2Fdocling | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

[![arXiv](https://img.shields.io/badge/arXiv-2408.09869-b31b1b.svg)](https://arxiv.org/abs/2408.09869)
[![Docs](https://img.shields.io/badge/docs-live-brightgreen)](https://docling-project.github.io/docling/)
[![PyPI version](https://img.shields.io/pypi/v/docling)](https://pypi.org/project/docling/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/docling)](https://pypi.org/project/docling/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pydantic v2](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/pydantic/pydantic/main/docs/badge/v2.json)](https://pydantic.dev)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![License MIT](https://img.shields.io/github/license/docling-project/docling)](https://opensource.org/licenses/MIT)
[![PyPI Downloads](https://static.pepy.tech/badge/docling/month)](https://pepy.tech/projects/docling)
[![Docling Actor](https://apify.com/actor-badge?actor=vancura/docling?fpr=docling)](https://apify.com/vancura/docling)
[![Chat with Dosu](https://dosu.dev/dosu-chat-badge.svg)](https://app.dosu.dev/097760a8-135e-4789-8234-90c8837d7f1c/ask?utm_source=github)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/10101/badge)](https://www.bestpractices.dev/projects/10101)
[![LF AI & Data](https://img.shields.io/badge/LF%20AI%20%26%20Data-003778?logo=linuxfoundation&logoColor=fff&color=0094ff&labelColor=003778)](https://lfaidata.foundation/projects/)

Docling simplifies document processing, parsing diverse formats — including advanced PDF understanding — and providing seamless integrations with the gen AI ecosystem.

## Features

* 🗂️ Parsing of [multiple document formats][supported_formats] incl. PDF, DOCX, PPTX, XLSX, HTML, WAV, MP3, images (PNG, TIFF, JPEG, ...), and more
* 📑 Advanced PDF understanding incl. page layout, reading order, table structure, code, formulas, image classification, and more
* 🧬 Unified, expressive [DoclingDocument][docling_document] representation format
* ↪️ Various [export formats][supported_formats] and options, including Markdown, HTML, [DocTags](https://arxiv.org/abs/2503.11576) and lossless JSON
* 🔒 Local execution capabilities for sensitive data and air-gapped environments
* 🤖 Plug-and-play [integrations][integrations] incl. LangChain, LlamaIndex, Crew AI & Haystack for agentic AI
* 🔍 Extensive OCR support for scanned PDFs and images
* 👓 Support of several Visual Language Models ([SmolDocling](https://huggingface.co/ds4sd/SmolDocling-256M-preview))
* 🎙️ Audio support with Automatic Speech Recognition (ASR) models
* 🔌 Connect to any agent using the [MCP server](https://docling-project.github.io/docling/usage/mcp/)
* 💻 Simple and convenient CLI

### What's new
* 📤 Structured [information extraction][extraction] \[🧪 beta\]
* 📑 New layout model (**Heron**) by default, for faster PDF parsing
* 🔌 [MCP server](https://docling-project.github.io/docling/usage/mcp/) for agentic applications

### Coming soon

* 📝 Metadata extraction, including title, authors, references & language
* 📝 Chart understanding (Barchart, Piechart, LinePlot, etc)
* 📝 Complex chemistry understanding (Molecular structures)
* 📝 Parsing of Web Video Text Tracks (WebVTT) files

## Installation

To use Docling, simply install `docling` from your package manager, e.g. pip:
```bash
pip install docling
```

Works on macOS, Linux and Windows environments. Both x86_64 and arm64 architectures.

More [detailed installation instructions](https://docling-project.github.io/docling/installation/) are available in the docs.

## Getting started

To convert individual documents with python, use `convert()`, for example:

```python
from docling.document_converter import DocumentConverter

source = "https://arxiv.org/pdf/2408.09869"  # document per local path or URL
converter = DocumentConverter()
result = converter.convert(source)
print(result.document.export_to_markdown())  # output: "## Docling Technical Report[...]"
```

More [advanced usage options](https://docling-project.github.io/docling/usage/advanced_options/) are available in
the docs.

## CLI

Docling has a built-in CLI to run conversions.

```bash
docling https://arxiv.org/pdf/2206.01062
```

You can also use 🥚[SmolDocling](https://huggingface.co/ds4sd/SmolDocling-256M-preview) and other VLMs via Docling CLI:
```bash
docling --pipeline vlm --vlm-model smoldocling https://arxiv.org/pdf/2206.01062
```
This will use MLX acceleration on supported Apple Silicon hardware.

Read more [here](https://docling-project.github.io/docling/usage/)

## Documentation

Check out Docling's [documentation](https://docling-project.github.io/docling/), for details on
installation, usage, concepts, recipes, extensions, and more.

## Examples

Go hands-on with our [examples](https://docling-project.github.io/docling/examples/),
demonstrating how to address different application use cases with Docling.

## Integrations

To further accelerate your AI application development, check out Docling's native
[integrations](https://docling-project.github.io/docling/integrations/) with popular frameworks
and tools.

## Get help and support

Please feel free to connect with us using the [discussion section](https://github.com/docling-project/docling/discussions).

## Technical report

For more details on Docling's inner workings, check out the [Docling Technical Report](https://arxiv.org/abs/2408.09869).

## Codebase Overview

The `docling` repository is structured into several key directories, each responsible for a specific part of the document processing lifecycle. The source code is located in the `docling/` directory.

*   **`docling/datamodel`**: Defines the core data structures used throughout the application. This includes the central `DoclingDocument` format, as well as configuration options for pipelines and models (e.g., `PipelineOptions`, `VlmModelSpecs`).

*   **`docling/pipeline`**: Contains the main processing pipelines that orchestrate the document conversion process. Pipelines like `StandardPdfPipeline` and `AsrPipeline` define the sequence of steps (e.g., OCR, layout analysis, text extraction) to convert a source file into a `DoclingDocument`.

*   **`docling/backend`**: Holds the different backends responsible for handling various input document formats. Each backend (e.g., `PdfBackend`, `DocxBackend`, `HtmlBackend`) knows how to parse a specific file type and perform an initial conversion.

*   **`docling/models`**: Implements the various models used for document understanding and enrichment. This includes a wide range of models for tasks such as Optical Character Recognition (OCR), layout analysis, table recognition, and Vision Language Model (VLM) processing.

*   **`docling/utils`**: Provides a collection of utility functions that support the core application logic. These utilities cover a range of functionalities, including model downloading, image processing, and data export.

*   **`docling/exceptions.py`**: Defines custom exception classes for tailored error handling within the application.

This structure separates concerns, making the codebase more modular and easier to navigate for contributors. For more detailed information, refer to the docstrings within each module.

## Contributing

Please read [Contributing to Docling](https://github.com/docling-project/docling/blob/main/CONTRIBUTING.md) for details.

## References

If you use Docling in your projects, please consider citing the following:

```bib
@techreport{Docling,
  author = {Deep Search Team},
  month = {8},
  title = {Docling Technical Report},
  url = {https://arxiv.org/abs/2408.09869},
  eprint = {2408.09869},
  doi = {10.48550/arXiv.2408.09869},
  version = {1.0.0},
  year = {2024}
}
```

## License

The Docling codebase is under MIT license.
For individual model usage, please refer to the model licenses found in the original packages.

## LF AI & Data

Docling is hosted as a project in the [LF AI & Data Foundation](https://lfaidata.foundation/projects/).

### IBM ❤️ Open Source AI

The project was started by the AI for knowledge team at IBM Research Zurich.

[supported_formats]: https://docling-project.github.io/docling/usage/supported_formats/
[docling_document]: https://docling-project.github.io/docling/concepts/docling_document/
[integrations]: https://docling-project.github.io/docling/integrations/
[extraction]: https://docling-project.github.io/docling/examples/extraction/
