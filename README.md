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
