# Root correction to the Megakernels reading interpretation

2026-10-12 22:55 UTC. Root read the new Megakernels transfer and scale-check sections and the complete current handoff. Preserve the original interpretation as history and cite this correction in the next bounded design record.

The arithmetic is a logical operation-count estimate. A 90–96% share of counted FLOPs does not show a 90–96% wall-time share, a compute bottleneck, or that launch latency is small. Remove the categorical conclusion “This block is dominated by large GEMMs, not by launch latency” from the future design's premises. Keep both possible bottlenecks unresolved until applicable timing evidence exists.

3072 output tiles relative to a plausible core count does not establish small wave-tail loss. Register pressure, occupancy, tile duration, scheduling and actual core count are unmeasured. Preserve the tile count as a structural fact only; it cannot rank the direction's measured headroom.

The 1% arithmetic is an illustrative target, not the acceptance floor or a causal falsifier. Recovering 1% through idle removal alone would require the stated approximate idle time; reducing materialization, changing overlap or other work does not follow that bound. Any positive stable gain remains eligible under its prospective contract. Missing a 1% idle target cannot close or reject a persistent scheduling design.

The limited CPU reading/design release remains active. No profiling or GPU diagnostic is authorized by this correction. Continue the already proposed Mirage reading, then compare designs with logical work, forecast costs and unknown measured shares explicitly separated. Keep calibration_usable=false, fc13 untouched, and the 00:38:25 UTC research cutoff / 00:58:25 UTC hard end.
