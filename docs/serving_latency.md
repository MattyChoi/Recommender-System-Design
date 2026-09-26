Target `localhost:50051`.

| offered rps | achieved rps | p50 ms | p95 ms | p99 ms | errors | fallback | degraded | worst source |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 100 | 100 | 7.9 | 9.9 | 11.3 | 0.00% | 0.00% | 0.00% | -- |
| 200 | 200 | 8.1 | 10.7 | 12.7 | 0.03% | 0.02% | 0.02% | ranker |
| 300 | 300 | 8.0 | 10.4 | 13.4 | 0.03% | 0.02% | 0.02% | two_tower |
| 400 | 400 | 8.6 | 29.8 | 31.9 **\*** | 0.09% | 1.01% | 1.01% | two_tower |
| 500 | 500 | 26.3 | 31.9 | 33.9 **\*** | 0.09% | 37.24% | 37.24% | two_tower |
| 750 | 750 | 26.6 | 27.0 | 27.3 **\*** | 0.09% | 99.76% | 99.76% | two_tower |
| 1,000 | 1,000 | 26.7 | 27.5 | 27.8 **\*** | 0.09% | 99.48% | 99.48% | two_tower |

**\*** saturated: p99 past 100 ms, throughput below the offered rate, or more than 1% of requests degraded.

`degraded` is the WORST SINGLE SOURCE's share, not a sum: one failure can degrade several sources at once -- a missed feature deadline takes the retriever with it, because the sidecar refuses a feature block of the wrong width -- so summing would double-count it.

Saturation first appears at **400 rps** (p99 31.9 ms, achieved 400) -- 1.0% of requests degraded (two_tower). That is the number the HPA in `infra/k8s` scales against.
