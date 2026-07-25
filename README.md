# Req2Inst

**Req2Inst: Toward Task Instruction Generation for Crowdsourcing from Multimodal Software Requirements**

Req2Inst is a research framework for transforming heterogeneous software requirements into clear and standardized task instructions for crowdsourced software development. It handles requirements expressed as text, images, and flowcharts, and converts them into a consistent instruction format intended to reduce ambiguity for crowdsourcing workers.

## Inputs and Output

Req2Inst considers three forms of requirement input:

- **Text requirements:** natural-language descriptions of intended software behavior.
- **Image requirements:** visual specifications such as user-interface screenshots and other image-based task contexts.
- **Flowchart requirements:** diagrams that describe procedural steps, execution flows, and logical dependencies.

Each generated instruction follows the same three-part structure:

- **Definition:** describes the task objective and expected output.
- **Emphasis & Caution:** highlights important constraints and considerations during task execution.
- **Things to Avoid:** identifies common mistakes and undesirable behaviors that should be prevented.

## Multimodal Preprocessing

The framework converts each input modality into a unified textual intermediate representation before instruction generation.

- Text requirements are condensed using TextRank-based sentence extraction and analyzed with dependency parsing to identify core functional elements and their relationships.
- Images are processed with BLIP-2 to extract visual entities, categories, and attributes and convert them into textual descriptions.
- Flowcharts are processed with Qwen-VL to identify key steps and sequential relationships and convert them into ordered textual descriptions that preserve logical dependencies.

This unified representation enables requirements from different modalities to be processed through a shared instruction-generation framework while retaining modality-specific semantic information.

## Multi-Expert LoRA Generation

Req2Inst uses Qwen3-8B as its backbone language model and applies parameter-efficient LoRA fine-tuning through four experts:

- **Text expert**
- **Image expert**
- **Flowchart expert**
- **General expert**, trained on mixed data

Each expert is an independent LoRA adapter trained on the same backbone. The experts specialize in the different textual distributions produced by modality-specific preprocessing.

A learned Router MLP assigns sample-level probabilities to the four experts. The manuscript describes two learned routing modes:

- **Learned top-1 routing:** selects the expert with the highest router probability for the input sample.
- **Learned top-2 output fusion:** selects the two highest-scoring experts and combines their token logits in the output space using normalized router-probability weights.

The top-2 strategy fuses model outputs rather than combining LoRA parameters, allowing multiple experts to contribute to ambiguous or mixed-domain requirements.

## Data Sources

The manuscript describes a multimodal requirement dataset assembled from the following sources:

- **Text:** 1,756 low-level requirements from publicly available software requirement datasets across computer science, medicine, and aerospace, including GANNT, WARC, CCHIT, InfusionPump, CM1, and MODIS.
- **Design2Code:** 500 user-interface design screenshots representing software-related visual specifications.
- **MS COCO:** 500 open-domain images representing general image-based crowdsourcing tasks.
- **Roboflow:** 1,500 flowchart images representing procedural requirements, execution processes, and logical dependencies.

## Repository layout

- `config/`: runtime paths and model, training, and inference settings.
- `models/`: language and vision model wrappers and prompt templates.
- `scripts/inference/`: command-line entrypoints for instruction generation and input recognition.
- `scripts/evaluation/`: metrics and experiment entrypoints.
- `src/`: instruction generation, routing, training, baselines, preprocessing, and shared utilities.
- `docs/`: reproducibility and local artifact guidance.
- `requirements.txt`: project dependency specification; it is not an exact environment lock.

## Installation

Create and activate an isolated Python environment, then install the dependencies from the repository root:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Use the PyTorch build appropriate for the target operating system and accelerator. The repository does not provide model files or trained checkpoints.

## Usage

The main entrypoint is `scripts/inference/generate_instructions.py`. After installing the dependencies, inspect its verified command-line interface with:

```bash
python scripts/inference/generate_instructions.py --help
```

To process the default local input structure and write JSON output:

```bash
python scripts/inference/generate_instructions.py --input-dir inputs --output-format json
```

The auxiliary recognition entrypoint accepts a local image or diagram file or directory:

```bash
python scripts/inference/recognize_inputs.py --help
python scripts/inference/recognize_inputs.py --input path/to/image.jpg --type image
```

These commands load the model runtime during startup. Actual inference requires compatible local models and any required local weights or checkpoints configured for the repository.

## Local inputs

The `inputs/` directory is local and ignored by Git. Use the following structure:

```text
inputs/
  text/   # .txt requirement files
  image/  # local .jpg or .png image requirements
  uml/    # local .jpg or .png diagram requirements
```

If `inputs/` does not exist, the main entrypoint creates these subdirectories and asks the user to add files. Do not commit confidential, copyrighted, or otherwise non-redistributable inputs.

## Data, models, and artifacts

Datasets, base models, model weights, checkpoints, user inputs, generated outputs, logs, and inference caches are local resources and are not distributed with this repository. They remain subject to their respective licenses and usage terms.

Generated files are written under `outputs/` and are ignored by Git. Cache-backed experiment modes require the corresponding predictions to be generated locally first. The repository does not claim that a source checkout alone provides a complete reproduction environment.

See [Data and artifact policy](docs/data-and-artifacts.md) and [Reproducibility guidance](docs/reproducibility.md).

## Documentation

The documentation index is available at [docs/README.md](docs/README.md).

## Citation

Citation metadata is provided in [CITATION.cff](CITATION.cff). It identifies the manuscript as a preferred citation without asserting publication status or a DOI.

## License

Req2Inst source code is licensed under the [Apache License 2.0](LICENSE). Third-party dependencies, models, datasets, checkpoints, inputs, and generated artifacts remain subject to their own licenses and terms.
