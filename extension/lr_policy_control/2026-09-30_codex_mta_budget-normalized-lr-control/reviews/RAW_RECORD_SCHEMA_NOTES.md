# Structural not-applicable alpha fields in preserved raw records

Operation: snd-lr-policy-verified-result-20261001. Source: genuine independent actual-result review 003, finding F3 (P3).

The 210 decay records contain 6,120 literal `NaN` tokens and the 30 bound fixed-policy reference records contain 1,800. They occur only in `alpha_mean` and `alpha_std`, which are not applicable to these softmax/top-k methods without a learnable alpha. There are no `Infinity` tokens. The six scientifically used epoch fields and all endpoints were independently checked as finite. This is a structural serialization issue, not a measured scientific failure or an imputed observation.

The frozen protocol's literal statement that completed rows must be finite is therefore not satisfied for every serialized field. That wording deviation is retained openly; it is not retrospectively rewritten or waived. The frozen analysis and independent review support only the finite scientific fields actually used in this control.

Python's permissive JSON decoder accepts these records; strict JSON parsers may reject the literal tokens. Custody bytes, hashes and original protocol/source files remain unchanged. A future public, strict-JSON export would need a separately versioned schema declaring these two fields not applicable, typically encoding them as `null`, and binding the transformation to the original record hashes. No such export or public release has been performed here. This note documents the existing raw format and closes the local documentation action only.
