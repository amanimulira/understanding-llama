# LLM ( large language models )


Highlights of Llama 4

- Mixture-of-Experts (MoE) Adoption:

MoEs are networks composed of "experts" sub-networks and a "gating" network that dynamically routes inputs to the appropriate experts. Allowing for conditional computation, thus large networks are made more efficient.

- Native Multimodality (Vision)

Provides deeper integration than "bolted-on" vision leading to better cross modal understanding and grounding.

Specifically, Early Fusion. MetaCLIP-based vision encoder generates visual tokens that are processed jointly with text tokens within the same Transformer backbone, hence cross-modal attention.

Requires joint pre-training on massive text/image/video datasets.

- Ultra-Long Context (iRoPE in Scout)

Interleaves standard RoPE attention layers with NoPE (No Positional Encoding) layers. Along with inference time temperature scaling and training on long sequences.

10 million token context window

- Advanced Post-Training Pipeline

Better balance between reasoning/coding capabilities and conversational alignment.

	> lightweight SFT focused on hard examples
	> intensive online RL focused on hard prompts using a dynamic curriculum and mixed-capability batches.
	> lightweight DPO for final polishing/corner cases. 

Also proactively embraces quantization (FP8, INT4) using optimized libraries like FBGEMM, making high-performance inference feasible.


