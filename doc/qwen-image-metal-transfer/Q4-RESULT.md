# First compressed-Q4 candidate

Opus authored the proposal; Astra performed peer review; root reviewed source and independently verified retained raw evidence on CPU.

All four cells passed the unchanged v5 protocol: two seeds and changed live inputs had zero recorded error, prior outputs survived, all native controls cleared the family alarm. Q4 setup audit and source review found no full dense weight preparation.

| Cell | Native ms | Candidate ms | Median reduction | p |
|---|---:|---:|---:|---:|
| t1024-text128 | 145.627865 | 143.963937 | 1.143% | 0.0009765625 |
| t1024-text512 | 148.217000 | 146.540063 | 1.131% | 0.001953125 |
| t2304-text128 | 366.964958 | 363.383052 | 0.976% | 0.0009765625 |
| t2304-text512 | 388.443135 | 383.832594 | 1.187% | 0.0009765625 |

Exact candidate: 5bcf49aa1c390559641a863c327defb8ed4e9c51 / fe6723d4cc692260c5ba6d4d599d202f634811ca70bca9dfb9f8a8d2593595f0. Selected as the first protocol-passing Q4 operator candidate; native wrapper remains rollback. These are complete block0 callable times with synthetic activations and real weights. Full-image/E2E qualification, profiling and causal attribution between concatenation and compilation remain unrun. Resource snapshots do not prove physical residency or thermal state. No main integration or publication.
