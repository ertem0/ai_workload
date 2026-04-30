# Workload Trace Format

This repository writes two trace artifacts during a benchmark run:

- `results/<experiment>/workload_trace.pkl`: the canonical workload trace.
- `results/<experiment>/expert_traces_raw.pkl`: a compact CWS routing sidecar used by the CWS analysis step.

For inspection, convert either pickle to JSON:

```bash
python scripts/workload_trace_to_json.py results/<experiment>/workload_trace.pkl
python scripts/pickle_trace_to_json.py results/<experiment>/expert_traces_raw.pkl
```

The JSON files are derived views for humans and downstream tools. Pickle files preserve the native Python payload and are what the pipeline writes first.

## `workload_trace.pkl`

`workload_trace.pkl` is a pickled Python `dict` with `schema_version: 1`. After JSON conversion, it has this top-level shape:

```json
{
  "schema_version": 1,
  "metadata": {},
  "model": {},
  "inferences": [],
  "routing_trace": {},
  "summary": {}
}
```

### `metadata`

Run-level context for interpreting the trace.

| Field | Type | Description |
| --- | --- | --- |
| `model_id` | string | Hugging Face model id from the experiment config. |
| `model_class` | string | Loaded `transformers` model class name. |
| `torch_version` | string | PyTorch version used for the run. |
| `transformers_version` | string | Transformers version used for the run. |
| `device` | string | Execution device recorded by the loader. |
| `requested_precision` | string or null | Precision requested in the config, such as `fp16`. |
| `eval_mode` | boolean | Whether the model was in eval mode when exported. |
| `use_cache` | boolean | Model config `use_cache` value at export time. |
| `configured_top_k` | integer or null | Router top-k used by the experiment. |
| `original_top_k` | integer or null | Router top-k from the original model config before override. |
| `routed_experts` | integer or null | Number of routed experts, excluding shared experts. |
| `created_at` | string | UTC ISO-8601 export timestamp. |
| `prompts_processed` | integer | Number of prompts in the resolved dataset. |
| `input_token_count` | integer | Total prompt tokens across prompts. |
| `output_token_count` | integer | Total generated tokens across prompts. |

### `model`

Static model inventory captured once per run.

`model.parameters` is an object keyed by full parameter name, for example `model.layers.0.self_attn.q_proj.weight`. Each value contains:

| Field | Type | Description |
| --- | --- | --- |
| `kind` | string | Always `parameter`. |
| `module` | string | Owning module path. |
| `tensor_name` | string | Local tensor name, such as `weight` or `bias`. |
| `shape` | integer array | Tensor dimensions. |
| `dtype` | string | Torch dtype string. |
| `numel` | integer | Number of elements. |
| `bytes` | integer | Storage size estimated as `numel * element_size`. |
| `requires_grad` | boolean | Native parameter flag. |
| `trainable` | boolean | Same value as `requires_grad`. |
| `static_during_inference` | boolean | Always `true` for parameters. |
| `role` | string | Coarse role inferred from the module path. |

`model.buffers` is keyed by full buffer name and uses the same tensor sizing fields, plus `persistent`.

`model.modules` is keyed by module path. Each value contains:

| Field | Type | Description |
| --- | --- | --- |
| `class` | string | Python class name of the module. |
| `role` | string | `module` for non-linear modules, or an inferred matrix role for linear modules. |
| `parameter_refs` | string array | Full parameter names owned directly by the module. |

`model.static_matrix_inventory` is a list of linear weight matrices that can be mapped to crossbars. Each entry contains:

| Field | Type | Description |
| --- | --- | --- |
| `matrix_id` | string | Full weight parameter name. |
| `module` | string | Owning module path. |
| `role` | string | Inferred matrix role. |
| `shape`, `rows`, `cols`, `dtype` | mixed | Matrix dimensions and dtype. |
| `static_during_inference` | boolean | Always `true`. |
| `crossbar` | object | Tiling information for the configured crossbar size. |

`crossbar` contains `tile_shape`, `tiles`, `used_cells`, `provisioned_cells`, and `tiling_efficiency`.

### `inferences`

One entry per profiled prompt prefill pass. Generation is decoded for user output, but AIMC operation tracing is currently attached to the forward prefill pass only.

Each inference contains:

| Field | Type | Description |
| --- | --- | --- |
| `inference_id` | integer | Zero-based trace id. |
| `prompt_index` | integer | One-based prompt index from the run. |
| `phase` | string | Currently `prefill`. |
| `input_shape` | object | Batch input tensor shapes, keyed by input name. |
| `input_dtypes` | object | Batch input tensor dtypes, keyed by input name. |
| `operations` | array | Runtime operation records captured by hooks. |
| `summary` | object | Per-inference aggregate counts and MAC estimates. |
| `routing` | array | Routing events for the same `prompt_index`. |

Operation records share `event_id`, `op_type`, `op_family`, `module`, and `role` when available. The remaining fields depend on operation family:

- `static_weight_matmul`: emitted for `nn.Linear`; includes activation `inputs`, static `weights`, `outputs`, and `math.macs`.
- `dynamic_activation_matmul`: emitted for attention matrix products such as `Q x K^T` and `Attention x V`; includes activation-only inputs and `math.macs`.
- `elementwise`: emitted for activation functions; includes input/output shapes and `element_count`.

Per-inference `summary` contains `total_ops`, `operation_counts`, `linear_ops`, `dynamic_matmul_ops`, `activation_ops`, `static_weight_macs`, `dynamic_activation_macs`, `nonlinear_element_ops`, and `activation_bytes`.

### Routing Records

Routing records appear in two places:

- `inferences[*].routing`: routing events attached to the corresponding inference.
- `routing_trace`: the same events grouped by prompt index.

In JSON, `routing_trace` keys are strings because JSON object keys are strings. In the pickle payload, they are integers.

Each routing record has:

| Field | Type | Description |
| --- | --- | --- |
| `prompt_index` | integer | One-based prompt index. |
| `phase` | string | `prefill` for prompt-token routing or `decode` for generated-token routing. |
| `token_position` | integer | Zero-based position in the full sequence. Decode positions start after the prompt length. |
| `token_id` | integer or null | Token id for the routed token. Decode token ids are filled after generation finalization when available. |
| `layer_id` | integer | Numeric transformer layer id, or `-1` if it cannot be inferred. |
| `layer_name` | string | Stable layer label such as `layer 0`. |
| `selected_experts` | integer array | Routed expert ids selected for that token-layer event. Length is usually `configured_top_k`. |

A routing event means "for this token at this layer, the router selected these top-k routed experts." Shared experts are not included.

### `summary`

Whole-trace aggregates for quick analysis:

| Field | Type | Description |
| --- | --- | --- |
| `num_inferences` | integer | Number of inference records. |
| `total_parameters` | integer | Count of parameter tensors. |
| `total_parameter_bytes` | integer | Sum of parameter byte estimates. |
| `total_static_matrices` | integer | Number of crossbar-mappable linear weights. |
| `total_runtime_ops` | integer | Count of captured operation records. |
| `total_static_weight_macs` | integer | Sum of linear/static-weight MAC estimates. |
| `total_dynamic_activation_macs` | integer | Sum of dynamic activation matmul MAC estimates. |
| `total_nonlinear_element_ops` | integer | Sum of elementwise activation element counts. |
| `operator_breakdown` | object | Captured operation counts by `op_type`. |
| `crossbar` | object | Global crossbar tiling totals and efficiency. |

## `expert_traces_raw.pkl`

`expert_traces_raw.pkl` is a smaller pickled `dict`:

```json
{
  "metadata": {},
  "routing_trace": {}
}
```

It is written whenever CWS is enabled and is the input to the phase-2 CWS analysis. The `routing_trace` records have the same structure as the workload trace routing records described above.

Its metadata usually includes `model_id`, `configured_top_k`, `original_top_k`, `execution_device`, `prompts_processed`, `input_token_count`, and `output_token_count`.

## Stability Notes

- `schema_version` applies to the workload trace payload. Increment it when adding, removing, or changing the meaning of fields.
- Tuple shapes in pickle become arrays after JSON conversion.
- Tensor-like values are normalized by the JSON conversion scripts into Python scalars or arrays.
- Paths under `results/<experiment>/` are run artifacts and may be regenerated. This document is the durable schema reference.
