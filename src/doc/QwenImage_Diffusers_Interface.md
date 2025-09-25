## Qwen-Image (diffusers) Interface Inventory and Call Flows

This note enumerates the Qwen-Image related modules inside `diffusers/` and summarizes key interfaces. Important parts are analyzed with brief comments. At the end, two arrow-style call-flow diagrams illustrate the main execution for Text-to-Image (T2I) and Image Editing (TI2I/I2I).

### 1) Transformers / Core Model

- `diffusers/src/diffusers/models/transformers/transformer_qwenimage.py`
  - **Classes**
    - `QwenImageTransformer2DModel`
      - Inputs: `hidden_states` (image stream features), `encoder_hidden_states` (text stream), `encoder_hidden_states_mask`, `timestep`, `img_shapes: List[(F,H,W)]`, `txt_seq_lens`, optional `guidance`, `attention_kwargs`
      - Output: `Transformer2DModelOutput(sample=...)`  // image-stream residuals after dual-stream blocks
      - Notes: dual-stream DiT; applies RoPE via `QwenEmbedRope`; time-text embedding via `QwenTimestepProjEmbeddings`.
    - `QwenImageTransformerBlock`
      - Performs joint attention (image/text) using `QwenDoubleStreamAttnProcessor2_0`; modulated norms/MLPs per stream; returns updated `(encoder_hidden_states, hidden_states)`
    - `QwenDoubleStreamAttnProcessor2_0`
      - Joint attention across concatenated text+image tokens; handles Q/K/V projection, QK norm, RoPE, concat, attention, split back, out projections
    - `QwenEmbedRope`, `QwenTimestepProjEmbeddings`
      - Build rotary embeddings for video-like (F,H,W) layout and sinusoidal timestep embeddings
  - **Key behavior**
    - Image path residual is projected to patch outputs at the end (`proj_out`)
    - ControlNet residuals can be fused into `hidden_states` loop

### 2) VAE and ControlNet

- `diffusers/src/diffusers/models/autoencoders/autoencoder_kl_qwenimage.py`
  - **Class**: `AutoencoderKLQwenImage`
    - Methods: `encode(image)` -> `latent_dist` or `latents`; `decode(latents)` -> pixels
    - Config fields: `latents_mean`, `latents_std`, latent channels count
  - **Used by**: modular encoder steps and final decode

- `diffusers/src/diffusers/models/controlnets/controlnet_qwenimage.py`
  - **Classes**: `QwenImageControlNetModel`, `QwenImageMultiControlNetModel`
    - Provide residuals for conditional control; multi variant accepts list of control images

### 3) Modular pipeline (recommended building blocks)

- `diffusers/src/diffusers/modular_pipelines/qwenimage/encoders.py`
  - **Text encoders**
    - `get_qwen_prompt_embeds(text_encoder, tokenizer, prompt, ...) -> (prompt_embeds, encoder_attention_mask)`
    - `get_qwen_prompt_embeds_edit(text_encoder, processor, prompt, image, ...) -> (prompt_embeds, encoder_attention_mask)`
      - Edit variant uses `Qwen2VLProcessor` and vision tokens in prompt template
  - **Image preprocess & VAE**
    - `QwenImageProcessImagesInputStep` / `QwenImageInpaintProcessImagesInputStep`
      - Preprocess PIL images (and mask for inpaint) to tensors
    - `QwenImageVaeEncoderDynamicStep`
      - `encode_vae_image(...)` wrapper to get normalized latents (subtract mean, divide std)
    - `QwenImageControlNetVaeEncoderStep`
      - Encodes control images into latents (single or list), with sampling mode
  - **Utilities**
    - `encode_vae_image(image, vae, generator, device, dtype, latent_channels, sample_mode)`
      - Returns normalized latents; shape can be 4D/5D depending on input

- `diffusers/src/diffusers/modular_pipelines/qwenimage/inputs.py`
  - Input dataclasses/typing for modular pipelines (pre/post structure)

- `diffusers/src/diffusers/modular_pipelines/qwenimage/decoders.py`
  - VAE decode steps and postprocess utilities

- `diffusers/src/diffusers/modular_pipelines/qwenimage/denoise.py`
  - Denoising scheduler orchestration around `QwenImageTransformer2DModel` (looping timesteps, guidance)

- `diffusers/src/diffusers/modular_pipelines/qwenimage/modular_pipeline.py`
  - `QwenImageModularPipeline`
    - Wires the blocks above into end-to-end pipelines (T2I, edit, inpaint, controlnet)

### 4) High-level pipelines (reference)

- `diffusers/src/diffusers/pipelines/qwenimage/pipeline_qwenimage.py`
  - Text-to-Image baseline; constructs embeddings, latents, denoise, decode
- `pipeline_qwenimage_img2img.py`
  - Image-to-Image (I2I) with given input image -> encode -> denoise -> decode
- `pipeline_qwenimage_inpaint.py`
  - Inpainting with mask -> preprocess (image/mask) -> encode -> denoise with masked guidance -> decode
- `pipeline_qwenimage_controlnet*.py`
  - Integrates controlnet latents into denoising path
- `pipeline_qwenimage_edit*.py`
  - Prompt+image editing variant using `get_qwen_prompt_embeds_edit`
- `pipeline_output.py`
  - Typed output containers for pipelines

### 5) Shared utils

- `diffusers/src/diffusers/pipelines/pipeline_utils.py`, `modular_pipelines/modular_pipeline.py`
  - Base classes for assembling components, arg validation, device plumbing

---

## Key Interfaces (I/O shapes)

- Text prompt embeddings
  - `prompt_embeds: FloatTensor [B, T_txt, D_txt]`
  - `encoder_attention_mask: Long/BoolTensor [B, T_txt]`

- Image latents (VAE)
  - `image_latents: FloatTensor [B, C_lat, H, W]` or `[B, C_lat, D, H, W]` (variant dependent)
  - Normalized by `latents_mean`/`latents_std`

- Control latents
  - `control_image_latents: FloatTensor` or `List[FloatTensor]` matching main latent layout

- Transformer forward
  - Inputs: `hidden_states` (image stream tokens), `encoder_hidden_states`, `encoder_hidden_states_mask`, `timestep`, `img_shapes`, `txt_seq_lens`, optional `guidance`
  - Output: `Transformer2DModelOutput(sample=...)`  // per-step residual (same tokenization layout as `hidden_states`)

- Decode
  - `pixels: FloatTensor [B, C, H, W]` after VAE decode and postprocess

---

## Call Flow Diagrams

### A) Text-to-Image (T2I)

```
Prompt (str)
  -> get_qwen_prompt_embeds(...)  -> prompt_embeds, prompt_mask
Image init (noise or distilled latents)
  -> QwenImageVaeEncoderDynamicStep (optional if starting from noise)
  -> image_latents

{prompt_embeds, prompt_mask, image_latents, img_shapes, txt_seq_lens, steps}
  -> Denoising loop (denoise.py)
     -> QwenImageTransformer2DModel.forward(..., timestep_t)
     -> residual_t -> update latents
  (repeat for T steps)

Final latents -> VAE.decode -> pixels -> (postprocess) -> Output
```

### B) Image Editing (TI2I / I2I / Inpaint)

```
Input image (+ optional mask/control images)
  -> QwenImageProcessImagesInputStep / InpaintProcessImagesInputStep
  -> QwenImageVaeEncoderDynamicStep / ControlNetVaeEncoderStep
  -> image_latents (+ control_image_latents) (+ processed_mask)

Prompt (str)
  -> get_qwen_prompt_embeds_edit(..., image=preprocessed_image)
  -> prompt_embeds, prompt_mask

{prompt_embeds, prompt_mask, image_latents, control_image_latents, img_shapes, txt_seq_lens, steps}
  -> Denoising loop (denoise.py)
     -> QwenImageTransformer2DModel.forward(..., timestep_t)
     -> fuse control residuals (if any)
     -> residual_t -> update (masked) latents
  (repeat for T steps)

Final latents -> VAE.decode -> pixels -> (apply mask overlay if inpaint) -> Output
```

---

## Notes for Integration (vLLM V1)

- Prefer modular pipeline blocks (encoders.py, denoise.py, decoders.py) to assemble inputs/loops.
- Keep shapes and dtype consistent; pass `img_shapes` and `txt_seq_lens` when using RoPE.
- For multi-outputs (pixels, latents, masks), define explicit conventions in the adapter or output schema.

---

## Function I/O Tables (Precise Signatures)

### encoders.py

#### get_qwen_prompt_embeds

| Item      | Description                                                                                                                                       |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| Signature | `get_qwen_prompt_embeds(text_encoder, tokenizer, prompt, prompt_template_encode, prompt_template_encode_start_idx, tokenizer_max_length, device)` |
| Inputs    | - **text_encoder**: `Qwen2_5_VLForConditionalGeneration` (HF model)  <br> - **tokenizer**: `Qwen2Tokenizer`  <br> - **prompt**: `str              | List[str]`  <br> - **prompt_template_encode**: `str` (template with `{}` placeholder)  <br> - **prompt_template_encode_start_idx**: `int`  <br> - **tokenizer_max_length**: `int`  <br> - **device**: `torch.device` |
| Outputs   | - **prompt_embeds**: `torch.FloatTensor [B, T_txt, D_txt]`  <br> - **encoder_attention_mask**: `torch.LongTensor/BoolTensor [B, T_txt]`           |
| Notes     | Applies template, tokenizes to `input_ids/attention_mask`, runs encoder, slices from `start_idx`, pads to max length in batch.                    |

#### get_qwen_prompt_embeds_edit

| Item      | Description                                                                                                                             |
| --------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| Signature | `get_qwen_prompt_embeds_edit(text_encoder, processor, prompt, image, prompt_template_encode, prompt_template_encode_start_idx, device)` |
| Inputs    | - **text_encoder**: `Qwen2_5_VLForConditionalGeneration`  <br> - **processor**: `Qwen2VLProcessor`  <br> - **prompt**: `str             | List[str]`  <br> - **image**: `torch.Tensor | PIL.Image | List[...]` (preprocessed by processor)  <br> - **prompt_template_encode**: `str`  <br> - **prompt_template_encode_start_idx**: `int`  <br> - **device**: `torch.device` |
| Outputs   | - **prompt_embeds**: `torch.FloatTensor [B, T_txt, D_txt]`  <br> - **encoder_attention_mask**: `torch.LongTensor/BoolTensor [B, T_txt]` |
| Notes     | Uses vision tokens via processor; otherwise similar slicing/padding as text-only.                                                       |

#### encode_vae_image

| Item      | Description                                                                                                                                                                             |
| --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Signature | `encode_vae_image(image, vae, generator, device, dtype, latent_channels=16, sample_mode="argmax")`                                                                                      |
| Inputs    | - **image**: `torch.Tensor [B, C, H, W]` or `[B, C, D, H, W]` (if 5D, function expects and preserves)  <br> - **vae**: `AutoencoderKLQwenImage`  <br> - **generator**: `torch.Generator | List[torch.Generator] | None`  <br> - **device**: `torch.device`  <br> - **dtype**: `torch.dtype`  <br> - **latent_channels**: `int`  <br> - **sample_mode**: `"sample" | "argmax"` |
| Outputs   | - **image_latents**: `torch.FloatTensor` normalized by `(latents - mean) / std`, shape `[B, latent_channels, ...]` (4D/5D)                                                              |
| Notes     | Pulls `vae.config.latents_mean/std`; handles per-item generators.                                                                                                                       |

#### QwenImageVaeEncoderDynamicStep.__call__

| Item      | Description                                                                                                                                                           |
| --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Signature | `__call__(components, state)`                                                                                                                                         |
| Inputs    | - **components.vae**: `AutoencoderKLQwenImage`  <br> - **state.processed_image**: `torch.Tensor` (from image processor)  <br> - **state.generator**: `torch.Generator | None`  <br> - **components._execution_device**: `torch.device`  <br> - **components.vae.dtype**: `torch.dtype` |
| Outputs   | - **state.image_latents**: `torch.FloatTensor` (normalized latents)  <br> - returns `(components, state)`                                                             |
| Notes     | Thin wrapper over `encode_vae_image`.                                                                                                                                 |

#### QwenImageControlNetVaeEncoderStep.__call__

| Item      | Description                                             |
| --------- | ------------------------------------------------------- |
| Signature | `__call__(components, state)`                           |
| Inputs    | - **components.controlnet**: `QwenImageControlNetModel  | QwenImageMultiControlNetModel`  <br> - **state.control_image**: image(s)  <br> - **height/width**: `int | None`  <br> - **components.vae** / processors |
| Outputs   | - **state.control_image_latents**: `torch.FloatTensor   | List[torch.FloatTensor]`  <br> - returns `(components, state)`                                          |
| Notes     | Preprocess then encode to latents (uses `sample` mode). |

### transformer_qwenimage.py

#### QwenImageTransformer2DModel.forward

| Item      | Description                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Signature | `forward(hidden_states, encoder_hidden_states=None, encoder_hidden_states_mask=None, timestep=None, img_shapes=None, txt_seq_lens=None, guidance=None, attention_kwargs=None, controlnet_block_samples=None, return_dict=True)`                                                                                                                                                                                                                 |
| Inputs    | - **hidden_states**: `torch.FloatTensor [B, S_img, D_in]` (image stream tokens)  <br> - **encoder_hidden_states**: `torch.FloatTensor [B, T_txt, D_txt]`  <br> - **encoder_hidden_states_mask**: `torch.LongTensor/BoolTensor [B, T_txt]`  <br> - **timestep**: `torch.LongTensor [B]`  <br> - **img_shapes**: `List[Tuple[int,int,int]]` (F,H,W per sample)  <br> - **txt_seq_lens**: `List[int]`  <br> - **guidance**: `torch.FloatTensor [B] | None`  <br> - **attention_kwargs**: `dict | None`  <br> - **controlnet_block_samples**: `Optional[List[torch.FloatTensor]]` |
| Outputs   | - if `return_dict=True`: `Transformer2DModelOutput(sample=FloatTensor [B, S_img, D_out])`  <br> - else: `Tuple(sample,)`                                                                                                                                                                                                                                                                                                                        |
| Notes     | Iterates `transformer_blocks`; fuses optional control residuals; final `norm_out + proj_out` on image path.                                                                                                                                                                                                                                                                                                                                     |

### pipelines (selected)

#### pipeline_qwenimage.py (call-style)

| Item    | Description                                                                               |
| ------- | ----------------------------------------------------------------------------------------- |
| Inputs  | - **prompt**: `str                                                                        | List[str]`  <br> - **num_inference_steps**: `int`  <br> - **guidance_scale**: `float`  <br> - **height/width**: `int`  <br> - optional seeds, processors |
| Outputs | - **images**: `List[PIL.Image.Image]` or tensors per pipeline config                      |
| Notes   | Orchestrates: text embeds -> init latents/noise -> denoise loop -> decode -> postprocess. |


