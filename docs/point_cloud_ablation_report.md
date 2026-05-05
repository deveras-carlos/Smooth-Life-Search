# Point-Cloud Ablation Report

Deterministic no-GIF ablation output. Lower best values are better.

## Primary Rosenbrock Cases

| case | variant | best value | delta vs default | ratio | evals | best source | active regions | coop active | improvements |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- | --- |
| rosenbrock_30d_seed7 | default | 2.554873e-03 | 0.000000e+00 | 1.000 | 6400 | direction_line_search | 0 | no | shade:331, region:252, cma:205 |
| rosenbrock_30d_seed7 | no_cooperative_refinement | 2.554873e-03 | 0.000000e+00 | 1.000 | 6400 | direction_line_search | 0 | no | shade:331, region:252, cma:205 |
| rosenbrock_30d_seed7 | no_coherent_probes | 1.969273e+01 | 1.969017e+01 | 7707.911 | 6400 | block_gradient | 0 | no | shade:570, region:320, cma:264 |
| rosenbrock_30d_seed7 | no_direction_refinement | 8.334278e-03 | 5.779406e-03 | 3.262 | 6400 | exploit_pattern | 1 | no | shade:489, region:351, cma:334 |
| rosenbrock_30d_seed7 | no_shade | 1.066654e-01 | 1.041105e-01 | 41.750 | 6400 | direction_line_search | 3 | no | cma:290, region:282, exploit:66 |
| rosenbrock_30d_seed7 | no_cma_region | 3.528258e-02 | 3.272771e-02 | 13.810 | 6400 | direction_line_search | 3 | no | shade:356, region:299, exploit:56 |
| rosenbrock_30d_seed7 | no_surrogate_ranking | 2.607686e+01 | 2.607430e+01 | 10206.715 | 6400 | exploit_pattern | 3 | no | shade:368, cma:275, region:262 |
| rosenbrock_30d_seed7 | no_restart_strategy | 2.603340e-01 | 2.577791e-01 | 101.897 | 6400 | direction_line_search | 2 | no | shade:468, region:284, cma:233 |
| rosenbrock_30d_seed7 | no_local_refinement | 2.636445e+01 | 2.636189e+01 | 10319.280 | 6400 | direction_line_search | 1 | no | shade:280, region:176, cma:137 |
| rosenbrock_30d_seed7 | evolution_only | 2.673610e+01 | 2.673354e+01 | 10464.749 | 6400 | exploit_stencil | 1 | no | shade:503, region:325, cma:257 |
| rosenbrock_30d_seed7 | polishing_only | 2.002608e+01 | 2.002353e+01 | 7838.388 | 6400 | block_gradient | 0 | no | region:480, exploit:167 |
| rosenbrock_50d_seed7 | default | 4.436251e-02 | 0.000000e+00 | 1.000 | 6400 | direction_line_search | 3 | no | shade:517, region:284, cma:267 |
| rosenbrock_50d_seed7 | no_cooperative_refinement | 4.436251e-02 | 0.000000e+00 | 1.000 | 6400 | direction_line_search | 3 | no | shade:517, region:284, cma:267 |
| rosenbrock_50d_seed7 | no_coherent_probes | 4.124971e+01 | 4.120535e+01 | 929.833 | 6400 | block_gradient | 0 | no | shade:775, region:367, cma:327 |
| rosenbrock_50d_seed7 | no_direction_refinement | 6.259542e-02 | 1.823291e-02 | 1.411 | 6400 | block_gradient | 0 | no | shade:649, region:367, cma:347 |
| rosenbrock_50d_seed7 | no_shade | 1.114978e-01 | 6.713529e-02 | 2.513 | 6400 | direction_line_search | 3 | no | region:389, cma:365, exploit:66 |
| rosenbrock_50d_seed7 | no_cma_region | 2.051641e-01 | 1.608016e-01 | 4.625 | 6400 | direction_line_search | 3 | no | shade:614, region:394, exploit:54 |
| rosenbrock_50d_seed7 | no_surrogate_ranking | 5.187665e-02 | 7.514140e-03 | 1.169 | 6400 | direction_line_search | 3 | no | shade:443, cma:271, region:256 |
| rosenbrock_50d_seed7 | no_restart_strategy | 1.162843e-01 | 7.192177e-02 | 2.621 | 6400 | direction_line_search | 1 | no | shade:589, region:333, cma:274 |
| rosenbrock_50d_seed7 | no_local_refinement | 4.732325e+01 | 4.727888e+01 | 1066.740 | 6400 | exploit_stencil | 0 | no | shade:382, region:190, cma:153 |
| rosenbrock_50d_seed7 | evolution_only | 4.714465e+01 | 4.710029e+01 | 1062.714 | 6400 | exploit_stencil | 3 | no | shade:593, region:289, cma:224 |
| rosenbrock_50d_seed7 | polishing_only | 4.156440e+01 | 4.152004e+01 | 936.926 | 6400 | exploit_pattern | 1 | no | region:632, exploit:148 |
| rosenbrock_100d_seed7 | default | 7.478288e-01 | 0.000000e+00 | 1.000 | 6400 | block_gradient | 1 | yes | shade:275, region:187, cma:137 |
| rosenbrock_100d_seed7 | no_cooperative_refinement | 1.851707e+00 | 1.103878e+00 | 2.476 | 6400 | direction_line_search | 1 | no | shade:470, region:266, cma:232 |
| rosenbrock_100d_seed7 | no_coherent_probes | 9.052289e+01 | 8.977506e+01 | 121.048 | 6400 | exploit_pattern | 0 | yes | shade:367, region:170, cma:144 |
| rosenbrock_100d_seed7 | no_direction_refinement | 1.998721e+00 | 1.250892e+00 | 2.673 | 6400 | block_gradient | 0 | yes | shade:317, region:205, cma:159 |
| rosenbrock_100d_seed7 | no_shade | 6.061535e-01 | -1.416753e-01 | 0.811 | 6400 | cooperative:line_search | 1 | yes | region:193, cma:171, cooperative:57 |
| rosenbrock_100d_seed7 | no_cma_region | 7.143417e-01 | -3.348713e-02 | 0.955 | 6400 | cooperative:line_search | 1 | yes | shade:331, region:201, cooperative:70 |
| rosenbrock_100d_seed7 | no_surrogate_ranking | 1.915233e+00 | 1.167404e+00 | 2.561 | 6400 | cooperative:line_search | 1 | yes | shade:284, region:164, cma:142 |
| rosenbrock_100d_seed7 | no_restart_strategy | 1.124988e+00 | 3.771593e-01 | 1.504 | 6400 | cooperative:line_search | 1 | yes | shade:290, region:154, cma:141 |
| rosenbrock_100d_seed7 | no_local_refinement | 9.603141e+01 | 9.528358e+01 | 128.414 | 6400 | exploit_stencil | 3 | yes | shade:225, cooperative:157, region:140 |
| rosenbrock_100d_seed7 | evolution_only | 1.563004e+01 | 1.488222e+01 | 20.901 | 6400 | exploit_stencil | 3 | no | shade:522, region:297, cma:250 |
| rosenbrock_100d_seed7 | polishing_only | 9.312621e+01 | 9.237838e+01 | 124.529 | 6400 | exploit_pattern | 0 | yes | region:310, exploit:66, cooperative:13 |
| rosenbrock_500d_seed7 | default | 1.883023e+01 | 0.000000e+00 | 1.000 | 6400 | direction_line_search | 1 | yes | shade:276, region:147, cma:117 |
| rosenbrock_500d_seed7 | no_cooperative_refinement | 4.891073e+02 | 4.702771e+02 | 25.975 | 6400 | block_gradient | 3 | no | shade:556, region:309, cma:246 |
| rosenbrock_500d_seed7 | no_coherent_probes | 4.896414e+02 | 4.708111e+02 | 26.003 | 6400 | block_gradient | 0 | yes | shade:370, region:229, cma:156 |
| rosenbrock_500d_seed7 | no_direction_refinement | 1.908811e+01 | 2.578768e-01 | 1.014 | 6400 | block_gradient | 0 | yes | shade:307, region:158, cma:129 |
| rosenbrock_500d_seed7 | no_shade | 2.191853e+01 | 3.088296e+00 | 1.164 | 6400 | direction_line_search | 1 | yes | region:205, cma:146, coherent:52 |
| rosenbrock_500d_seed7 | no_cma_region | 1.988179e+01 | 1.051564e+00 | 1.056 | 6400 | direction_line_search | 1 | yes | shade:320, region:194, cooperative:67 |
| rosenbrock_500d_seed7 | no_surrogate_ranking | 2.095625e+01 | 2.126021e+00 | 1.113 | 6400 | block_gradient | 1 | yes | shade:288, region:154, cma:131 |
| rosenbrock_500d_seed7 | no_restart_strategy | 2.026478e+01 | 1.434554e+00 | 1.076 | 6400 | cooperative:line_search | 1 | yes | shade:331, region:162, cma:113 |
| rosenbrock_500d_seed7 | no_local_refinement | 4.918992e+02 | 4.730690e+02 | 26.123 | 6400 | exploit_stencil | 3 | yes | shade:225, cooperative:187, region:125 |
| rosenbrock_500d_seed7 | evolution_only | 4.939903e+02 | 4.751600e+02 | 26.234 | 6400 | exploit_stencil | 3 | no | shade:530, region:277, cma:194 |
| rosenbrock_500d_seed7 | polishing_only | 4.962181e+02 | 4.773879e+02 | 26.352 | 6400 | block_gradient | 0 | yes | region:333, exploit:53, cooperative:21 |

## Guardrail Summary

| group | variant | cases | median ratio vs default | exact diagonal scout hits |
| --- | --- | ---: | ---: | ---: |
| transformed_12d | default | 21 | 1.000 | 0 |
| transformed_12d | no_cooperative_refinement | 21 | 1.000 | 0 |
| transformed_12d | no_coherent_probes | 21 | 1.272 | 0 |
| transformed_12d | no_direction_refinement | 21 | 0.814 | 0 |
| transformed_12d | no_shade | 21 | 1.155 | 0 |
| transformed_12d | no_cma_region | 21 | 0.785 | 0 |
| transformed_12d | no_surrogate_ranking | 21 | 1.264 | 0 |
| transformed_12d | no_restart_strategy | 21 | 0.646 | 0 |
| transformed_12d | no_local_refinement | 21 | 1.417 | 0 |
| transformed_12d | evolution_only | 21 | 0.223 | 0 |
| transformed_12d | polishing_only | 21 | 0.995 | 0 |
| anchor | default | 3 | n/a | 0 |
| anchor | no_cooperative_refinement | 3 | n/a | 0 |
| anchor | no_coherent_probes | 3 | n/a | 0 |
| anchor | no_direction_refinement | 3 | n/a | 0 |
| anchor | no_shade | 3 | n/a | 0 |
| anchor | no_cma_region | 3 | n/a | 0 |
| anchor | no_surrogate_ranking | 3 | n/a | 0 |
| anchor | no_restart_strategy | 3 | n/a | 0 |
| anchor | no_local_refinement | 3 | n/a | 0 |
| anchor | evolution_only | 3 | n/a | 0 |
| anchor | polishing_only | 3 | n/a | 0 |
| stress | default | 1 | n/a | 0 |
| stress | no_cooperative_refinement | 1 | n/a | 0 |
| stress | no_coherent_probes | 1 | n/a | 0 |
| stress | no_direction_refinement | 1 | n/a | 0 |
| stress | no_shade | 1 | n/a | 0 |
| stress | no_cma_region | 1 | n/a | 0 |
| stress | no_surrogate_ranking | 1 | n/a | 0 |
| stress | no_restart_strategy | 1 | n/a | 0 |
| stress | no_local_refinement | 1 | n/a | 0 |
| stress | evolution_only | 1 | n/a | 0 |
| stress | polishing_only | 1 | n/a | 0 |

## Notes

- `delta vs default` is positive when the ablated run is worse than the default for the same case and seed.
- Symmetric origin-anchor cases are guardrails only; exact zero there confirms anchor behavior, not search quality.
- `n/a` ratios occur when the default reaches exact zero, as in origin-anchor and shifted Ackley stress runs.
- `exact diagonal scout hits` must stay zero for high-D Rosenbrock-style shortcut protection.
