# The moe_out site tests and zeroes the MoE output alone

The `moe_out` zeroing site selects values by the magnitude of a layer's MoE
output h (the router-weighted expert sum) on its own, zeroes only h, and leaves
the residual stream r bit-identical. It was requested as "block-residual", but
`block` already means the stream sum h + r and vLLM's `residual` is r, the skip
stream this site must not touch; hence the new name. Testing h alone makes it the
direct counterpart of `block` for asking whether block zeroing's damage comes from
wiping the accumulated stream or from sparsifying one layer's update, and it gives
a token row of the same width (2048) so quantile arms at equal q zero equal shares
at both sites.

## Considered Options

- Test h + r (as `block`) but zero only h: selection would track the stream's
  scale rather than the update's, mixing the two effects the comparison separates.
- Name it `block_residual`: collides with both the glossary's `block` and vLLM's
  `residual`.
