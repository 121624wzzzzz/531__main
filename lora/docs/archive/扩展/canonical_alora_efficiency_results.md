# Results: canonical_alora_efficiency_20260525_231722

| run | variant | rank | trainable | eval_loss | Δ vs baseline | train_runtime_s |
|---|---|---:|---:|---:|---:|---:|
| qwen25_05b__seed42__affine_inlmh_s1_16 | affine_input_lm_head | 16 | 58k | 1.638 | 0.000 | 386 |
| qwen25_05b__seed42__affine_input_s1_64 | affine_input | 16 | 30k | 1.783 | 0.145 | 384 |
| qwen25_05b__seed42__affine_lm_head_s1_64 | affine_lm_head | 16 | 29k | 1.854 | 0.216 | 243 |
| qwen25_05b__seed42__mergeable_tied_affine_rank32_s1_16 | affine_input_lm_head | 16 | 58k | 1.639 | 0.001 | 394 |
| qwen25_05b__seed42__single_q_l0_r16 | hidden_lora | 16 | 29k | 2.128 | 0.490 | 382 |
| qwen25_05b__seed42__single_q_l12_r16 | hidden_lora | 16 | 29k | 2.034 | 0.396 | 309 |
| qwen25_05b__seed42__single_q_l23_r16 | hidden_lora | 16 | 29k | 2.135 | 0.497 | 244 |
| qwen25_05b__seed42__single_qkvo_l12_r4 | hidden_lora | 4 | 23k | 1.854 | 0.216 | 312 |
| qwen25_05b__seed42__vocab_only_r1 | hidden_lora | 1 | 306k | 1.928 | 0.290 | 409 |
| qwen25_05b__seed42__vocab_only_r2 | hidden_lora | 2 | 611k | 1.880 | 0.242 | 406 |
| qwen25_05b__seed43__affine_inlmh_s1_16 | affine_input_lm_head | 16 | 58k | 1.637 | -0.001 | 379 |
| qwen25_05b__seed43__affine_input_s1_64 | affine_input | 16 | 30k | 1.781 | 0.143 | 379 |
| qwen25_05b__seed43__affine_lm_head_s1_64 | affine_lm_head | 16 | 29k | 1.852 | 0.214 | 239 |
| qwen25_05b__seed43__mergeable_tied_affine_rank32_s1_16 | affine_input_lm_head | 16 | 58k | 1.636 | -0.002 | 390 |
| qwen25_05b__seed43__single_q_l0_r16 | hidden_lora | 16 | 29k | 2.125 | 0.487 | 378 |
| qwen25_05b__seed43__single_q_l12_r16 | hidden_lora | 16 | 29k | 2.032 | 0.394 | 305 |
| qwen25_05b__seed43__single_q_l23_r16 | hidden_lora | 16 | 29k | 2.135 | 0.497 | 243 |
| qwen25_05b__seed43__single_qkvo_l12_r4 | hidden_lora | 4 | 23k | 1.851 | 0.213 | 310 |
| qwen25_05b__seed43__vocab_only_r1 | hidden_lora | 1 | 306k | 1.928 | 0.290 | 404 |
| qwen25_05b__seed43__vocab_only_r2 | hidden_lora | 2 | 611k | 1.879 | 0.241 | 405 |
| qwen25_05b__seed44__affine_inlmh_s1_16 | affine_input_lm_head | 16 | 58k | 1.638 | 0.000 | 381 |
| qwen25_05b__seed44__affine_input_s1_64 | affine_input | 16 | 30k | 1.781 | 0.143 | 378 |
| qwen25_05b__seed44__affine_lm_head_s1_64 | affine_lm_head | 16 | 29k | 1.855 | 0.217 | 239 |
| qwen25_05b__seed44__mergeable_tied_affine_rank32_s1_16 | affine_input_lm_head | 16 | 58k | 1.638 | 0.000 | 391 |
| qwen25_05b__seed44__single_q_l0_r16 | hidden_lora | 16 | 29k | 2.126 | 0.488 | 378 |
| qwen25_05b__seed44__single_q_l12_r16 | hidden_lora | 16 | 29k | 2.033 | 0.395 | 308 |
| qwen25_05b__seed44__single_q_l23_r16 | hidden_lora | 16 | 29k | 2.135 | 0.497 | 244 |
| qwen25_05b__seed44__single_qkvo_l12_r4 | hidden_lora | 4 | 23k | 1.854 | 0.216 | 312 |
| qwen25_05b__seed44__vocab_only_r1 | hidden_lora | 1 | 306k | 1.929 | 0.291 | 405 |
| qwen25_05b__seed44__vocab_only_r2 | hidden_lora | 2 | 611k | 1.878 | 0.240 | 405 |
| qwen25_15b__seed42__affine_inlmh_s1_16 | affine_input_lm_head | 16 | 100k | 1.324 | -0.314 | 697 |
| qwen25_15b__seed42__affine_input_s1_64 | affine_input | 16 | 51k | 1.477 | -0.161 | 697 |
| qwen25_15b__seed42__affine_lm_head_s1_64 | affine_lm_head | 16 | 49k | 1.556 | -0.082 | 400 |
| qwen25_15b__seed42__mergeable_tied_affine_rank32_s1_16 | affine_input_lm_head | 16 | 100k | 1.317 | -0.321 | 705 |
| qwen25_15b__seed42__single_q_l0_r16 | hidden_lora | 16 | 49k | 1.774 | 0.136 | 694 |
| qwen25_15b__seed42__single_q_l14_r16 | hidden_lora | 16 | 49k | 1.722 | 0.084 | 548 |
| qwen25_15b__seed42__single_q_l27_r16 | hidden_lora | 16 | 49k | 1.851 | 0.213 | 412 |
| qwen25_15b__seed42__single_qkvo_l14_r4 | hidden_lora | 4 | 39k | 1.492 | -0.146 | 552 |
| qwen25_15b__seed42__vocab_only_r1 | hidden_lora | 1 | 307k | 1.623 | -0.015 | 718 |
| qwen25_15b__seed42__vocab_only_r2 | hidden_lora | 2 | 614k | 1.592 | -0.046 | 719 |
| qwen25_15b__seed43__affine_inlmh_s1_16 | affine_input_lm_head | 16 | 100k | 1.328 | -0.310 | 692 |
| qwen25_15b__seed43__affine_input_s1_64 | affine_input | 16 | 51k | 1.476 | -0.162 | 691 |
| qwen25_15b__seed43__affine_lm_head_s1_64 | affine_lm_head | 16 | 49k | 1.556 | -0.082 | 396 |
| qwen25_15b__seed43__mergeable_tied_affine_rank32_s1_16 | affine_input_lm_head | 16 | 100k | 1.322 | -0.316 | 706 |
| qwen25_15b__seed43__single_q_l0_r16 | hidden_lora | 16 | 49k | 1.775 | 0.137 | 694 |
| qwen25_15b__seed43__single_q_l14_r16 | hidden_lora | 16 | 49k | 1.720 | 0.082 | 544 |
| qwen25_15b__seed43__single_q_l27_r16 | hidden_lora | 16 | 49k | 1.852 | 0.214 | 409 |
| qwen25_15b__seed43__single_qkvo_l14_r4 | hidden_lora | 4 | 39k | 1.488 | -0.150 | 554 |
| qwen25_15b__seed43__vocab_only_r1 | hidden_lora | 1 | 307k | 1.629 | -0.009 | 715 |
| qwen25_15b__seed43__vocab_only_r2 | hidden_lora | 2 | 614k | 1.595 | -0.043 | 719 |
| qwen25_15b__seed44__affine_inlmh_s1_16 | affine_input_lm_head | 16 | 100k | 1.324 | -0.314 | 695 |
| qwen25_15b__seed44__affine_input_s1_64 | affine_input | 16 | 51k | 1.465 | -0.173 | 697 |
| qwen25_15b__seed44__affine_lm_head_s1_64 | affine_lm_head | 16 | 49k | 1.556 | -0.082 | 401 |
| qwen25_15b__seed44__mergeable_tied_affine_rank32_s1_16 | affine_input_lm_head | 16 | 100k | 1.318 | -0.320 | 706 |
| qwen25_15b__seed44__single_q_l0_r16 | hidden_lora | 16 | 49k | 1.772 | 0.134 | 690 |
| qwen25_15b__seed44__single_q_l14_r16 | hidden_lora | 16 | 49k | 1.721 | 0.083 | 550 |
| qwen25_15b__seed44__single_q_l27_r16 | hidden_lora | 16 | 49k | 1.849 | 0.211 | 412 |
| qwen25_15b__seed44__single_qkvo_l14_r4 | hidden_lora | 4 | 39k | 1.479 | -0.159 | 555 |
| qwen25_15b__seed44__vocab_only_r1 | hidden_lora | 1 | 307k | 1.628 | -0.010 | 720 |
| qwen25_15b__seed44__vocab_only_r2 | hidden_lora | 2 | 614k | 1.586 | -0.052 | 719 |
| qwen25_7b__seed42__affine_inlmh_s1_16 | affine_input_lm_head | 16 | 233k | 1.115 | -0.523 | 2442 |
| qwen25_7b__seed42__affine_input_s1_64 | affine_input | 16 | 118k | 1.230 | -0.408 | 2452 |
| qwen25_7b__seed42__affine_lm_head_s1_64 | affine_lm_head | 16 | 115k | 1.350 | -0.288 | 2464 |
| qwen25_7b__seed42__single_q_l0_r16 | hidden_lora | 16 | 115k | 1.553 | -0.085 | 2443 |
| qwen25_7b__seed42__single_q_l14_r16 | hidden_lora | 16 | 115k | 1.513 | -0.125 | 2423 |
| qwen25_7b__seed42__single_q_l27_r16 | hidden_lora | 16 | 115k | 1.618 | -0.020 | 2461 |
| qwen25_7b__seed42__single_qkvo_l14_r4 | hidden_lora | 4 | 90k | 1.253 | -0.385 | 2432 |
| qwen25_7b__seed42__vocab_only_r1 | hidden_lora | 1 | 311k | 1.460 | -0.178 | 2441 |
| qwen25_7b__seed42__vocab_only_r2 | hidden_lora | 2 | 623k | 1.427 | -0.211 | 2474 |
| qwen25_7b__seed43__affine_inlmh_s1_16 | affine_input_lm_head | 16 | 233k | 1.121 | -0.517 | 2419 |
| qwen25_7b__seed43__affine_input_s1_64 | affine_input | 16 | 118k | 1.229 | -0.409 | 2442 |
| qwen25_7b__seed43__affine_lm_head_s1_64 | affine_lm_head | 16 | 115k | 1.350 | -0.288 | 2457 |
| qwen25_7b__seed43__single_q_l0_r16 | hidden_lora | 16 | 115k | 1.554 | -0.084 | 2444 |
| qwen25_7b__seed43__single_q_l14_r16 | hidden_lora | 16 | 115k | 1.515 | -0.123 | 2410 |
| qwen25_7b__seed43__single_q_l27_r16 | hidden_lora | 16 | 115k | 1.619 | -0.019 | 2450 |
| qwen25_7b__seed43__single_qkvo_l14_r4 | hidden_lora | 4 | 90k | 1.251 | -0.387 | 2424 |
| qwen25_7b__seed43__vocab_only_r1 | hidden_lora | 1 | 311k | 1.460 | -0.178 | 2452 |
| qwen25_7b__seed43__vocab_only_r2 | hidden_lora | 2 | 623k | 1.418 | -0.220 | 2460 |
| qwen25_7b__seed44__affine_inlmh_s1_16 | affine_input_lm_head | 16 | 233k | 1.119 | -0.519 | 2459 |
| qwen25_7b__seed44__affine_input_s1_64 | affine_input | 16 | 118k | 1.228 | -0.410 | 2416 |
| qwen25_7b__seed44__affine_lm_head_s1_64 | affine_lm_head | 16 | 115k | 1.348 | -0.290 | 2462 |
| qwen25_7b__seed44__single_q_l0_r16 | hidden_lora | 16 | 115k | 1.550 | -0.088 | 2457 |
| qwen25_7b__seed44__single_q_l14_r16 | hidden_lora | 16 | 115k | 1.517 | -0.121 | 2443 |
| qwen25_7b__seed44__single_q_l27_r16 | hidden_lora | 16 | 115k | 1.617 | -0.021 | 2433 |
| qwen25_7b__seed44__single_qkvo_l14_r4 | hidden_lora | 4 | 90k | 1.256 | -0.382 | 2430 |
| qwen25_7b__seed44__vocab_only_r1 | hidden_lora | 1 | 311k | 1.458 | -0.180 | 2444 |
| qwen25_7b__seed44__vocab_only_r2 | hidden_lora | 2 | 623k | 1.422 | -0.216 | 2440 |
| qwen3_06b__seed42__affine_inlmh_s1_16 | affine_input_lm_head | 16 | 67k | 1.325 | -0.313 | 632 |
| qwen3_06b__seed42__affine_input_s1_64 | affine_input | 16 | 34k | 1.420 | -0.218 | 630 |
| qwen3_06b__seed42__affine_lm_head_s1_64 | affine_lm_head | 16 | 33k | 1.569 | -0.069 | 337 |
| qwen3_06b__seed42__mergeable_tied_affine_rank32_s1_16 | affine_input_lm_head | 16 | 67k | 1.323 | -0.315 | 644 |
| qwen3_06b__seed42__single_q_l0_r16 | hidden_lora | 16 | 49k | 1.626 | -0.012 | 627 |
| qwen3_06b__seed42__single_q_l14_r16 | hidden_lora | 16 | 49k | 1.668 | 0.030 | 482 |
| qwen3_06b__seed42__single_q_l27_r16 | hidden_lora | 16 | 49k | 1.781 | 0.143 | 346 |
| qwen3_06b__seed42__single_qkvo_l14_r4 | hidden_lora | 4 | 41k | 1.473 | -0.165 | 488 |
| qwen3_06b__seed42__vocab_only_r1 | hidden_lora | 1 | 306k | 1.569 | -0.069 | 656 |
| qwen3_06b__seed42__vocab_only_r2 | hidden_lora | 2 | 612k | 1.549 | -0.089 | 659 |
| qwen3_06b__seed43__affine_inlmh_s1_16 | affine_input_lm_head | 16 | 67k | 1.325 | -0.313 | 630 |
| qwen3_06b__seed43__affine_input_s1_64 | affine_input | 16 | 34k | 1.418 | -0.220 | 628 |
| qwen3_06b__seed43__affine_lm_head_s1_64 | affine_lm_head | 16 | 33k | 1.568 | -0.070 | 336 |
| qwen3_06b__seed43__mergeable_tied_affine_rank32_s1_16 | affine_input_lm_head | 16 | 67k | 1.323 | -0.315 | 641 |
| qwen3_06b__seed43__single_q_l0_r16 | hidden_lora | 16 | 49k | 1.629 | -0.009 | 624 |
| qwen3_06b__seed43__single_q_l14_r16 | hidden_lora | 16 | 49k | 1.667 | 0.029 | 480 |
| qwen3_06b__seed43__single_q_l27_r16 | hidden_lora | 16 | 49k | 1.779 | 0.141 | 342 |
| qwen3_06b__seed43__single_qkvo_l14_r4 | hidden_lora | 4 | 41k | 1.477 | -0.161 | 486 |
| qwen3_06b__seed43__vocab_only_r1 | hidden_lora | 1 | 306k | 1.570 | -0.068 | 655 |
| qwen3_06b__seed43__vocab_only_r2 | hidden_lora | 2 | 612k | 1.549 | -0.089 | 656 |
| qwen3_06b__seed44__affine_inlmh_s1_16 | affine_input_lm_head | 16 | 67k | 1.325 | -0.313 | 632 |
| qwen3_06b__seed44__affine_input_s1_64 | affine_input | 16 | 34k | 1.420 | -0.218 | 630 |
| qwen3_06b__seed44__affine_lm_head_s1_64 | affine_lm_head | 16 | 33k | 1.570 | -0.068 | 337 |
| qwen3_06b__seed44__mergeable_tied_affine_rank32_s1_16 | affine_input_lm_head | 16 | 67k | 1.321 | -0.317 | 641 |
| qwen3_06b__seed44__single_q_l0_r16 | hidden_lora | 16 | 49k | 1.630 | -0.008 | 628 |
| qwen3_06b__seed44__single_q_l14_r16 | hidden_lora | 16 | 49k | 1.667 | 0.029 | 480 |
| qwen3_06b__seed44__single_q_l27_r16 | hidden_lora | 16 | 49k | 1.780 | 0.142 | 345 |
| qwen3_06b__seed44__single_qkvo_l14_r4 | hidden_lora | 4 | 41k | 1.477 | -0.161 | 485 |
| qwen3_06b__seed44__vocab_only_r1 | hidden_lora | 1 | 306k | 1.572 | -0.066 | 658 |
| qwen3_06b__seed44__vocab_only_r2 | hidden_lora | 2 | 612k | 1.539 | -0.099 | 658 |
| qwen3_17b__seed42__affine_inlmh_s1_16 | affine_input_lm_head | 16 | 133k | 0.912 | -0.726 | 889 |
| qwen3_17b__seed42__affine_input_s1_64 | affine_input | 16 | 68k | 0.981 | -0.657 | 890 |
| qwen3_17b__seed42__affine_lm_head_s1_64 | affine_lm_head | 16 | 66k | 1.185 | -0.453 | 484 |
| qwen3_17b__seed42__mergeable_tied_affine_rank32_s1_16 | affine_input_lm_head | 16 | 133k | 0.911 | -0.727 | 896 |
| qwen3_17b__seed42__single_q_l0_r16 | hidden_lora | 16 | 66k | 1.273 | -0.365 | 880 |
| qwen3_17b__seed42__single_q_l14_r16 | hidden_lora | 16 | 66k | 1.273 | -0.365 | 684 |
| qwen3_17b__seed42__single_q_l27_r16 | hidden_lora | 16 | 66k | 1.395 | -0.243 | 493 |
| qwen3_17b__seed42__single_qkvo_l14_r4 | hidden_lora | 4 | 57k | 1.058 | -0.580 | 693 |
| qwen3_17b__seed42__vocab_only_r1 | hidden_lora | 1 | 308k | 1.226 | -0.412 | 910 |
| qwen3_17b__seed42__vocab_only_r2 | hidden_lora | 2 | 616k | 1.199 | -0.439 | 909 |
| qwen3_17b__seed43__affine_inlmh_s1_16 | affine_input_lm_head | 16 | 133k | 0.909 | -0.729 | 886 |
| qwen3_17b__seed43__affine_input_s1_64 | affine_input | 16 | 68k | 0.979 | -0.659 | 885 |
| qwen3_17b__seed43__affine_lm_head_s1_64 | affine_lm_head | 16 | 66k | 1.183 | -0.455 | 482 |
| qwen3_17b__seed43__mergeable_tied_affine_rank32_s1_16 | affine_input_lm_head | 16 | 133k | 0.911 | -0.727 | 899 |
| qwen3_17b__seed43__single_q_l0_r16 | hidden_lora | 16 | 66k | 1.269 | -0.369 | 881 |
| qwen3_17b__seed43__single_q_l14_r16 | hidden_lora | 16 | 66k | 1.269 | -0.369 | 684 |
| qwen3_17b__seed43__single_q_l27_r16 | hidden_lora | 16 | 66k | 1.395 | -0.243 | 490 |
| qwen3_17b__seed43__single_qkvo_l14_r4 | hidden_lora | 4 | 57k | 1.055 | -0.583 | 691 |
| qwen3_17b__seed43__vocab_only_r1 | hidden_lora | 1 | 308k | 1.232 | -0.406 | 909 |
| qwen3_17b__seed43__vocab_only_r2 | hidden_lora | 2 | 616k | 1.189 | -0.449 | 910 |
| qwen3_17b__seed44__affine_inlmh_s1_16 | affine_input_lm_head | 16 | 133k | 0.914 | -0.724 | 892 |
| qwen3_17b__seed44__affine_input_s1_64 | affine_input | 16 | 68k | 0.980 | -0.658 | 883 |
| qwen3_17b__seed44__affine_lm_head_s1_64 | affine_lm_head | 16 | 66k | 1.184 | -0.454 | 482 |
| qwen3_17b__seed44__mergeable_tied_affine_rank32_s1_16 | affine_input_lm_head | 16 | 133k | 0.907 | -0.731 | 896 |
| qwen3_17b__seed44__single_q_l0_r16 | hidden_lora | 16 | 66k | 1.277 | -0.361 | 885 |
| qwen3_17b__seed44__single_q_l14_r16 | hidden_lora | 16 | 66k | 1.270 | -0.368 | 686 |
| qwen3_17b__seed44__single_q_l27_r16 | hidden_lora | 16 | 66k | 1.394 | -0.244 | 494 |
| qwen3_17b__seed44__single_qkvo_l14_r4 | hidden_lora | 4 | 57k | 1.057 | -0.581 | 691 |
| qwen3_17b__seed44__vocab_only_r1 | hidden_lora | 1 | 308k | 1.205 | -0.433 | 910 |
| qwen3_17b__seed44__vocab_only_r2 | hidden_lora | 2 | 616k | 1.192 | -0.446 | 910 |
| qwen3_8b__seed42__affine_inlmh_s1_16 | affine_input_lm_head | 16 | 266k | 1.012 | -0.626 | 3079 |
| qwen3_8b__seed42__affine_input_s1_64 | affine_input | 16 | 135k | 1.067 | -0.571 | 3097 |
| qwen3_8b__seed42__affine_lm_head_s1_64 | affine_lm_head | 16 | 131k | 1.201 | -0.437 | 3069 |
| qwen3_8b__seed42__single_q_l0_r16 | hidden_lora | 16 | 131k | 1.345 | -0.293 | 3054 |
| qwen3_8b__seed42__single_q_l18_r16 | hidden_lora | 16 | 131k | 1.306 | -0.332 | 3099 |
| qwen3_8b__seed42__single_q_l35_r16 | hidden_lora | 16 | 131k | 1.377 | -0.261 | 3057 |
| qwen3_8b__seed42__single_qkvo_l18_r4 | hidden_lora | 4 | 106k | 1.124 | -0.514 | 3067 |
| qwen3_8b__seed42__vocab_only_r1 | hidden_lora | 1 | 312k | 1.293 | -0.345 | 3110 |
| qwen3_8b__seed42__vocab_only_r2 | hidden_lora | 2 | 624k | 1.281 | -0.357 | 3094 |
| qwen3_8b__seed43__affine_inlmh_s1_16 | affine_input_lm_head | 16 | 266k | 1.010 | -0.628 | 3044 |
| qwen3_8b__seed43__affine_input_s1_64 | affine_input | 16 | 135k | 1.069 | -0.569 | 3044 |
| qwen3_8b__seed43__affine_lm_head_s1_64 | affine_lm_head | 16 | 131k | 1.202 | -0.436 | 3084 |
| qwen3_8b__seed43__single_q_l0_r16 | hidden_lora | 16 | 131k | 1.356 | -0.282 | 3083 |
| qwen3_8b__seed43__single_q_l18_r16 | hidden_lora | 16 | 131k | 1.308 | -0.330 | 3080 |
| qwen3_8b__seed43__single_q_l35_r16 | hidden_lora | 16 | 131k | 1.380 | -0.258 | 3055 |
| qwen3_8b__seed43__single_qkvo_l18_r4 | hidden_lora | 4 | 106k | 1.123 | -0.515 | 3072 |
| qwen3_8b__seed43__vocab_only_r1 | hidden_lora | 1 | 312k | 1.287 | -0.351 | 3099 |
| qwen3_8b__seed43__vocab_only_r2 | hidden_lora | 2 | 624k | 1.276 | -0.362 | 3072 |
| qwen3_8b__seed44__affine_inlmh_s1_16 | affine_input_lm_head | 16 | 266k | 1.009 | -0.629 | 3090 |
| qwen3_8b__seed44__affine_input_s1_64 | affine_input | 16 | 135k | 1.064 | -0.574 | 3095 |
| qwen3_8b__seed44__affine_lm_head_s1_64 | affine_lm_head | 16 | 131k | 1.201 | -0.437 | 3078 |
| qwen3_8b__seed44__single_q_l0_r16 | hidden_lora | 16 | 131k | 1.346 | -0.292 | 3076 |
| qwen3_8b__seed44__single_q_l18_r16 | hidden_lora | 16 | 131k | 1.308 | -0.330 | 3062 |
| qwen3_8b__seed44__single_q_l35_r16 | hidden_lora | 16 | 131k | 1.378 | -0.260 | 3053 |
| qwen3_8b__seed44__single_qkvo_l18_r4 | hidden_lora | 4 | 106k | 1.119 | -0.519 | 3064 |
| qwen3_8b__seed44__vocab_only_r1 | hidden_lora | 1 | 312k | 1.289 | -0.349 | 3082 |
| qwen3_8b__seed44__vocab_only_r2 | hidden_lora | 2 | 624k | 1.277 | -0.361 | 3082 |

```json
[
  {
    "run_id": "qwen25_05b__seed42__affine_inlmh_s1_16",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 256.0,
    "seed": 42,
    "trainable": 58240,
    "total": 494091008,
    "pct": 0.011787302148190482,
    "eval_loss": 1.638,
    "train_runtime_s": 386.3,
    "train_loss": 1.695
  },
  {
    "run_id": "qwen25_05b__seed42__affine_input_s1_64",
    "variant": "affine_input",
    "hidden_lora_rank": 16,
    "affine_alpha": 1024.0,
    "seed": 42,
    "trainable": 29568,
    "total": 494062336,
    "pct": 0.005984669918250964,
    "eval_loss": 1.783,
    "train_runtime_s": 384.4,
    "train_loss": 1.835
  },
  {
    "run_id": "qwen25_05b__seed42__affine_lm_head_s1_64",
    "variant": "affine_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 1024.0,
    "seed": 42,
    "trainable": 28672,
    "total": 494061440,
    "pct": 0.005803326808908625,
    "eval_loss": 1.854,
    "train_runtime_s": 242.7,
    "train_loss": 1.898
  },
  {
    "run_id": "qwen25_05b__seed42__mergeable_tied_affine_rank32_s1_16",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 512.0,
    "seed": 42,
    "trainable": 58240,
    "total": 494091008,
    "pct": 0.011787302148190482,
    "eval_loss": 1.639,
    "train_runtime_s": 393.9,
    "train_loss": 1.698
  },
  {
    "run_id": "qwen25_05b__seed42__single_q_l0_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 28672,
    "total": 494061440,
    "pct": 0.005803326808908625,
    "eval_loss": 2.128,
    "train_runtime_s": 382.5,
    "train_loss": 2.157
  },
  {
    "run_id": "qwen25_05b__seed42__single_q_l12_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 28672,
    "total": 494061440,
    "pct": 0.005803326808908625,
    "eval_loss": 2.034,
    "train_runtime_s": 309.4,
    "train_loss": 2.074
  },
  {
    "run_id": "qwen25_05b__seed42__single_q_l23_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 28672,
    "total": 494061440,
    "pct": 0.005803326808908625,
    "eval_loss": 2.135,
    "train_runtime_s": 243.9,
    "train_loss": 2.165
  },
  {
    "run_id": "qwen25_05b__seed42__single_qkvo_l12_r4",
    "variant": "hidden_lora",
    "hidden_lora_rank": 4,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 22528,
    "total": 494055296,
    "pct": 0.004559813482902125,
    "eval_loss": 1.854,
    "train_runtime_s": 312.4,
    "train_loss": 1.91
  },
  {
    "run_id": "qwen25_05b__seed42__vocab_only_r1",
    "variant": "hidden_lora",
    "hidden_lora_rank": 1,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 305664,
    "total": 494338432,
    "pct": 0.061832942820840606,
    "eval_loss": 1.928,
    "train_runtime_s": 408.9,
    "train_loss": 1.962
  },
  {
    "run_id": "qwen25_05b__seed42__vocab_only_r2",
    "variant": "hidden_lora",
    "hidden_lora_rank": 2,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 611328,
    "total": 494644096,
    "pct": 0.12358946663744269,
    "eval_loss": 1.88,
    "train_runtime_s": 406.3,
    "train_loss": 1.924
  },
  {
    "run_id": "qwen25_05b__seed43__affine_inlmh_s1_16",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 256.0,
    "seed": 43,
    "trainable": 58240,
    "total": 494091008,
    "pct": 0.011787302148190482,
    "eval_loss": 1.637,
    "train_runtime_s": 378.7,
    "train_loss": 1.694
  },
  {
    "run_id": "qwen25_05b__seed43__affine_input_s1_64",
    "variant": "affine_input",
    "hidden_lora_rank": 16,
    "affine_alpha": 1024.0,
    "seed": 43,
    "trainable": 29568,
    "total": 494062336,
    "pct": 0.005984669918250964,
    "eval_loss": 1.781,
    "train_runtime_s": 379.1,
    "train_loss": 1.832
  },
  {
    "run_id": "qwen25_05b__seed43__affine_lm_head_s1_64",
    "variant": "affine_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 1024.0,
    "seed": 43,
    "trainable": 28672,
    "total": 494061440,
    "pct": 0.005803326808908625,
    "eval_loss": 1.852,
    "train_runtime_s": 239.4,
    "train_loss": 1.897
  },
  {
    "run_id": "qwen25_05b__seed43__mergeable_tied_affine_rank32_s1_16",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 512.0,
    "seed": 43,
    "trainable": 58240,
    "total": 494091008,
    "pct": 0.011787302148190482,
    "eval_loss": 1.636,
    "train_runtime_s": 390.2,
    "train_loss": 1.697
  },
  {
    "run_id": "qwen25_05b__seed43__single_q_l0_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 43,
    "trainable": 28672,
    "total": 494061440,
    "pct": 0.005803326808908625,
    "eval_loss": 2.125,
    "train_runtime_s": 378.4,
    "train_loss": 2.155
  },
  {
    "run_id": "qwen25_05b__seed43__single_q_l12_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 43,
    "trainable": 28672,
    "total": 494061440,
    "pct": 0.005803326808908625,
    "eval_loss": 2.032,
    "train_runtime_s": 305.4,
    "train_loss": 2.074
  },
  {
    "run_id": "qwen25_05b__seed43__single_q_l23_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 43,
    "trainable": 28672,
    "total": 494061440,
    "pct": 0.005803326808908625,
    "eval_loss": 2.135,
    "train_runtime_s": 243.3,
    "train_loss": 2.165
  },
  {
    "run_id": "qwen25_05b__seed43__single_qkvo_l12_r4",
    "variant": "hidden_lora",
    "hidden_lora_rank": 4,
    "affine_alpha": 32.0,
    "seed": 43,
    "trainable": 22528,
    "total": 494055296,
    "pct": 0.004559813482902125,
    "eval_loss": 1.851,
    "train_runtime_s": 309.9,
    "train_loss": 1.907
  },
  {
    "run_id": "qwen25_05b__seed43__vocab_only_r1",
    "variant": "hidden_lora",
    "hidden_lora_rank": 1,
    "affine_alpha": 32.0,
    "seed": 43,
    "trainable": 305664,
    "total": 494338432,
    "pct": 0.061832942820840606,
    "eval_loss": 1.928,
    "train_runtime_s": 404.1,
    "train_loss": 1.964
  },
  {
    "run_id": "qwen25_05b__seed43__vocab_only_r2",
    "variant": "hidden_lora",
    "hidden_lora_rank": 2,
    "affine_alpha": 32.0,
    "seed": 43,
    "trainable": 611328,
    "total": 494644096,
    "pct": 0.12358946663744269,
    "eval_loss": 1.879,
    "train_runtime_s": 404.7,
    "train_loss": 1.925
  },
  {
    "run_id": "qwen25_05b__seed44__affine_inlmh_s1_16",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 256.0,
    "seed": 44,
    "trainable": 58240,
    "total": 494091008,
    "pct": 0.011787302148190482,
    "eval_loss": 1.638,
    "train_runtime_s": 380.6,
    "train_loss": 1.693
  },
  {
    "run_id": "qwen25_05b__seed44__affine_input_s1_64",
    "variant": "affine_input",
    "hidden_lora_rank": 16,
    "affine_alpha": 1024.0,
    "seed": 44,
    "trainable": 29568,
    "total": 494062336,
    "pct": 0.005984669918250964,
    "eval_loss": 1.781,
    "train_runtime_s": 378.0,
    "train_loss": 1.832
  },
  {
    "run_id": "qwen25_05b__seed44__affine_lm_head_s1_64",
    "variant": "affine_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 1024.0,
    "seed": 44,
    "trainable": 28672,
    "total": 494061440,
    "pct": 0.005803326808908625,
    "eval_loss": 1.855,
    "train_runtime_s": 239.3,
    "train_loss": 1.898
  },
  {
    "run_id": "qwen25_05b__seed44__mergeable_tied_affine_rank32_s1_16",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 512.0,
    "seed": 44,
    "trainable": 58240,
    "total": 494091008,
    "pct": 0.011787302148190482,
    "eval_loss": 1.638,
    "train_runtime_s": 391.3,
    "train_loss": 1.699
  },
  {
    "run_id": "qwen25_05b__seed44__single_q_l0_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 44,
    "trainable": 28672,
    "total": 494061440,
    "pct": 0.005803326808908625,
    "eval_loss": 2.126,
    "train_runtime_s": 378.2,
    "train_loss": 2.155
  },
  {
    "run_id": "qwen25_05b__seed44__single_q_l12_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 44,
    "trainable": 28672,
    "total": 494061440,
    "pct": 0.005803326808908625,
    "eval_loss": 2.033,
    "train_runtime_s": 307.5,
    "train_loss": 2.074
  },
  {
    "run_id": "qwen25_05b__seed44__single_q_l23_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 44,
    "trainable": 28672,
    "total": 494061440,
    "pct": 0.005803326808908625,
    "eval_loss": 2.135,
    "train_runtime_s": 243.8,
    "train_loss": 2.165
  },
  {
    "run_id": "qwen25_05b__seed44__single_qkvo_l12_r4",
    "variant": "hidden_lora",
    "hidden_lora_rank": 4,
    "affine_alpha": 32.0,
    "seed": 44,
    "trainable": 22528,
    "total": 494055296,
    "pct": 0.004559813482902125,
    "eval_loss": 1.854,
    "train_runtime_s": 311.9,
    "train_loss": 1.91
  },
  {
    "run_id": "qwen25_05b__seed44__vocab_only_r1",
    "variant": "hidden_lora",
    "hidden_lora_rank": 1,
    "affine_alpha": 32.0,
    "seed": 44,
    "trainable": 305664,
    "total": 494338432,
    "pct": 0.061832942820840606,
    "eval_loss": 1.929,
    "train_runtime_s": 405.1,
    "train_loss": 1.965
  },
  {
    "run_id": "qwen25_05b__seed44__vocab_only_r2",
    "variant": "hidden_lora",
    "hidden_lora_rank": 2,
    "affine_alpha": 32.0,
    "seed": 44,
    "trainable": 611328,
    "total": 494644096,
    "pct": 0.12358946663744269,
    "eval_loss": 1.878,
    "train_runtime_s": 405.2,
    "train_loss": 1.922
  },
  {
    "run_id": "qwen25_15b__seed42__affine_inlmh_s1_16",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 256.0,
    "seed": 42,
    "trainable": 99840,
    "total": 1543814144,
    "pct": 0.0064670997080850685,
    "eval_loss": 1.324,
    "train_runtime_s": 696.9,
    "train_loss": 1.379
  },
  {
    "run_id": "qwen25_15b__seed42__affine_input_s1_64",
    "variant": "affine_input",
    "hidden_lora_rank": 16,
    "affine_alpha": 1024.0,
    "seed": 42,
    "trainable": 50688,
    "total": 1543764992,
    "pct": 0.003283401311901235,
    "eval_loss": 1.477,
    "train_runtime_s": 696.8,
    "train_loss": 1.528
  },
  {
    "run_id": "qwen25_15b__seed42__affine_lm_head_s1_64",
    "variant": "affine_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 1024.0,
    "seed": 42,
    "trainable": 49152,
    "total": 1543763456,
    "pct": 0.0031839074703423993,
    "eval_loss": 1.556,
    "train_runtime_s": 400.1,
    "train_loss": 1.601
  },
  {
    "run_id": "qwen25_15b__seed42__mergeable_tied_affine_rank32_s1_16",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 512.0,
    "seed": 42,
    "trainable": 99840,
    "total": 1543814144,
    "pct": 0.0064670997080850685,
    "eval_loss": 1.317,
    "train_runtime_s": 705.4,
    "train_loss": 1.376
  },
  {
    "run_id": "qwen25_15b__seed42__single_q_l0_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 49152,
    "total": 1543763456,
    "pct": 0.0031839074703423993,
    "eval_loss": 1.774,
    "train_runtime_s": 694.5,
    "train_loss": 1.808
  },
  {
    "run_id": "qwen25_15b__seed42__single_q_l14_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 49152,
    "total": 1543763456,
    "pct": 0.0031839074703423993,
    "eval_loss": 1.722,
    "train_runtime_s": 548.1,
    "train_loss": 1.765
  },
  {
    "run_id": "qwen25_15b__seed42__single_q_l27_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 49152,
    "total": 1543763456,
    "pct": 0.0031839074703423993,
    "eval_loss": 1.851,
    "train_runtime_s": 411.7,
    "train_loss": 1.88
  },
  {
    "run_id": "qwen25_15b__seed42__single_qkvo_l14_r4",
    "variant": "hidden_lora",
    "hidden_lora_rank": 4,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 38912,
    "total": 1543753216,
    "pct": 0.0025206101335823873,
    "eval_loss": 1.492,
    "train_runtime_s": 552.3,
    "train_loss": 1.554
  },
  {
    "run_id": "qwen25_15b__seed42__vocab_only_r1",
    "variant": "hidden_lora",
    "hidden_lora_rank": 1,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 306944,
    "total": 1544021248,
    "pct": 0.019879519170969334,
    "eval_loss": 1.623,
    "train_runtime_s": 718.0,
    "train_loss": 1.661
  },
  {
    "run_id": "qwen25_15b__seed42__vocab_only_r2",
    "variant": "hidden_lora",
    "hidden_lora_rank": 2,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 613888,
    "total": 1544328192,
    "pct": 0.039751136007235434,
    "eval_loss": 1.592,
    "train_runtime_s": 718.9,
    "train_loss": 1.634
  },
  {
    "run_id": "qwen25_15b__seed43__affine_inlmh_s1_16",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 256.0,
    "seed": 43,
    "trainable": 99840,
    "total": 1543814144,
    "pct": 0.0064670997080850685,
    "eval_loss": 1.328,
    "train_runtime_s": 692.3,
    "train_loss": 1.383
  },
  {
    "run_id": "qwen25_15b__seed43__affine_input_s1_64",
    "variant": "affine_input",
    "hidden_lora_rank": 16,
    "affine_alpha": 1024.0,
    "seed": 43,
    "trainable": 50688,
    "total": 1543764992,
    "pct": 0.003283401311901235,
    "eval_loss": 1.476,
    "train_runtime_s": 691.4,
    "train_loss": 1.529
  },
  {
    "run_id": "qwen25_15b__seed43__affine_lm_head_s1_64",
    "variant": "affine_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 1024.0,
    "seed": 43,
    "trainable": 49152,
    "total": 1543763456,
    "pct": 0.0031839074703423993,
    "eval_loss": 1.556,
    "train_runtime_s": 396.5,
    "train_loss": 1.602
  },
  {
    "run_id": "qwen25_15b__seed43__mergeable_tied_affine_rank32_s1_16",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 512.0,
    "seed": 43,
    "trainable": 99840,
    "total": 1543814144,
    "pct": 0.0064670997080850685,
    "eval_loss": 1.322,
    "train_runtime_s": 706.2,
    "train_loss": 1.379
  },
  {
    "run_id": "qwen25_15b__seed43__single_q_l0_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 43,
    "trainable": 49152,
    "total": 1543763456,
    "pct": 0.0031839074703423993,
    "eval_loss": 1.775,
    "train_runtime_s": 693.5,
    "train_loss": 1.807
  },
  {
    "run_id": "qwen25_15b__seed43__single_q_l14_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 43,
    "trainable": 49152,
    "total": 1543763456,
    "pct": 0.0031839074703423993,
    "eval_loss": 1.72,
    "train_runtime_s": 544.2,
    "train_loss": 1.764
  },
  {
    "run_id": "qwen25_15b__seed43__single_q_l27_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 43,
    "trainable": 49152,
    "total": 1543763456,
    "pct": 0.0031839074703423993,
    "eval_loss": 1.852,
    "train_runtime_s": 409.4,
    "train_loss": 1.881
  },
  {
    "run_id": "qwen25_15b__seed43__single_qkvo_l14_r4",
    "variant": "hidden_lora",
    "hidden_lora_rank": 4,
    "affine_alpha": 32.0,
    "seed": 43,
    "trainable": 38912,
    "total": 1543753216,
    "pct": 0.0025206101335823873,
    "eval_loss": 1.488,
    "train_runtime_s": 554.1,
    "train_loss": 1.553
  },
  {
    "run_id": "qwen25_15b__seed43__vocab_only_r1",
    "variant": "hidden_lora",
    "hidden_lora_rank": 1,
    "affine_alpha": 32.0,
    "seed": 43,
    "trainable": 306944,
    "total": 1544021248,
    "pct": 0.019879519170969334,
    "eval_loss": 1.629,
    "train_runtime_s": 715.2,
    "train_loss": 1.666
  },
  {
    "run_id": "qwen25_15b__seed43__vocab_only_r2",
    "variant": "hidden_lora",
    "hidden_lora_rank": 2,
    "affine_alpha": 32.0,
    "seed": 43,
    "trainable": 613888,
    "total": 1544328192,
    "pct": 0.039751136007235434,
    "eval_loss": 1.595,
    "train_runtime_s": 718.7,
    "train_loss": 1.638
  },
  {
    "run_id": "qwen25_15b__seed44__affine_inlmh_s1_16",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 256.0,
    "seed": 44,
    "trainable": 99840,
    "total": 1543814144,
    "pct": 0.0064670997080850685,
    "eval_loss": 1.324,
    "train_runtime_s": 694.7,
    "train_loss": 1.378
  },
  {
    "run_id": "qwen25_15b__seed44__affine_input_s1_64",
    "variant": "affine_input",
    "hidden_lora_rank": 16,
    "affine_alpha": 1024.0,
    "seed": 44,
    "trainable": 50688,
    "total": 1543764992,
    "pct": 0.003283401311901235,
    "eval_loss": 1.465,
    "train_runtime_s": 697.4,
    "train_loss": 1.52
  },
  {
    "run_id": "qwen25_15b__seed44__affine_lm_head_s1_64",
    "variant": "affine_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 1024.0,
    "seed": 44,
    "trainable": 49152,
    "total": 1543763456,
    "pct": 0.0031839074703423993,
    "eval_loss": 1.556,
    "train_runtime_s": 401.0,
    "train_loss": 1.601
  },
  {
    "run_id": "qwen25_15b__seed44__mergeable_tied_affine_rank32_s1_16",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 512.0,
    "seed": 44,
    "trainable": 99840,
    "total": 1543814144,
    "pct": 0.0064670997080850685,
    "eval_loss": 1.318,
    "train_runtime_s": 705.9,
    "train_loss": 1.376
  },
  {
    "run_id": "qwen25_15b__seed44__single_q_l0_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 44,
    "trainable": 49152,
    "total": 1543763456,
    "pct": 0.0031839074703423993,
    "eval_loss": 1.772,
    "train_runtime_s": 690.1,
    "train_loss": 1.805
  },
  {
    "run_id": "qwen25_15b__seed44__single_q_l14_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 44,
    "trainable": 49152,
    "total": 1543763456,
    "pct": 0.0031839074703423993,
    "eval_loss": 1.721,
    "train_runtime_s": 549.8,
    "train_loss": 1.765
  },
  {
    "run_id": "qwen25_15b__seed44__single_q_l27_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 44,
    "trainable": 49152,
    "total": 1543763456,
    "pct": 0.0031839074703423993,
    "eval_loss": 1.849,
    "train_runtime_s": 411.7,
    "train_loss": 1.879
  },
  {
    "run_id": "qwen25_15b__seed44__single_qkvo_l14_r4",
    "variant": "hidden_lora",
    "hidden_lora_rank": 4,
    "affine_alpha": 32.0,
    "seed": 44,
    "trainable": 38912,
    "total": 1543753216,
    "pct": 0.0025206101335823873,
    "eval_loss": 1.479,
    "train_runtime_s": 555.3,
    "train_loss": 1.544
  },
  {
    "run_id": "qwen25_15b__seed44__vocab_only_r1",
    "variant": "hidden_lora",
    "hidden_lora_rank": 1,
    "affine_alpha": 32.0,
    "seed": 44,
    "trainable": 306944,
    "total": 1544021248,
    "pct": 0.019879519170969334,
    "eval_loss": 1.628,
    "train_runtime_s": 720.2,
    "train_loss": 1.664
  },
  {
    "run_id": "qwen25_15b__seed44__vocab_only_r2",
    "variant": "hidden_lora",
    "hidden_lora_rank": 2,
    "affine_alpha": 32.0,
    "seed": 44,
    "trainable": 613888,
    "total": 1544328192,
    "pct": 0.039751136007235434,
    "eval_loss": 1.586,
    "train_runtime_s": 719.0,
    "train_loss": 1.628
  },
  {
    "run_id": "qwen25_7b__seed42__affine_inlmh_s1_16",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 256.0,
    "seed": 42,
    "trainable": 232960,
    "total": 7615849472,
    "pct": 0.0030588839873541026,
    "eval_loss": 1.115,
    "train_runtime_s": 2442.0,
    "train_loss": 1.17
  },
  {
    "run_id": "qwen25_7b__seed42__affine_input_s1_64",
    "variant": "affine_input",
    "hidden_lora_rank": 16,
    "affine_alpha": 1024.0,
    "seed": 42,
    "trainable": 118272,
    "total": 7615734784,
    "pct": 0.0015529952572466053,
    "eval_loss": 1.23,
    "train_runtime_s": 2452.0,
    "train_loss": 1.278
  },
  {
    "run_id": "qwen25_7b__seed42__affine_lm_head_s1_64",
    "variant": "affine_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 1024.0,
    "seed": 42,
    "trainable": 114688,
    "total": 7615731200,
    "pct": 0.0015059355036060095,
    "eval_loss": 1.35,
    "train_runtime_s": 2464.0,
    "train_loss": 1.394
  },
  {
    "run_id": "qwen25_7b__seed42__single_q_l0_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 114688,
    "total": 7615731200,
    "pct": 0.0015059355036060095,
    "eval_loss": 1.553,
    "train_runtime_s": 2443.0,
    "train_loss": 1.587
  },
  {
    "run_id": "qwen25_7b__seed42__single_q_l14_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 114688,
    "total": 7615731200,
    "pct": 0.0015059355036060095,
    "eval_loss": 1.513,
    "train_runtime_s": 2423.0,
    "train_loss": 1.548
  },
  {
    "run_id": "qwen25_7b__seed42__single_q_l27_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 114688,
    "total": 7615731200,
    "pct": 0.0015059355036060095,
    "eval_loss": 1.618,
    "train_runtime_s": 2461.0,
    "train_loss": 1.645
  },
  {
    "run_id": "qwen25_7b__seed42__single_qkvo_l14_r4",
    "variant": "hidden_lora",
    "hidden_lora_rank": 4,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 90112,
    "total": 7615706624,
    "pct": 0.0011832388568648728,
    "eval_loss": 1.253,
    "train_runtime_s": 2432.0,
    "train_loss": 1.305
  },
  {
    "run_id": "qwen25_7b__seed42__vocab_only_r1",
    "variant": "hidden_lora",
    "hidden_lora_rank": 1,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 311296,
    "total": 7615927808,
    "pct": 0.004087433702732913,
    "eval_loss": 1.46,
    "train_runtime_s": 2441.0,
    "train_loss": 1.488
  },
  {
    "run_id": "qwen25_7b__seed42__vocab_only_r2",
    "variant": "hidden_lora",
    "hidden_lora_rank": 2,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 622592,
    "total": 7616239104,
    "pct": 0.008174533276837627,
    "eval_loss": 1.427,
    "train_runtime_s": 2474.0,
    "train_loss": 1.461
  },
  {
    "run_id": "qwen25_7b__seed43__affine_inlmh_s1_16",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 256.0,
    "seed": 43,
    "trainable": 232960,
    "total": 7615849472,
    "pct": 0.0030588839873541026,
    "eval_loss": 1.121,
    "train_runtime_s": 2419.0,
    "train_loss": 1.174
  },
  {
    "run_id": "qwen25_7b__seed43__affine_input_s1_64",
    "variant": "affine_input",
    "hidden_lora_rank": 16,
    "affine_alpha": 1024.0,
    "seed": 43,
    "trainable": 118272,
    "total": 7615734784,
    "pct": 0.0015529952572466053,
    "eval_loss": 1.229,
    "train_runtime_s": 2442.0,
    "train_loss": 1.28
  },
  {
    "run_id": "qwen25_7b__seed43__affine_lm_head_s1_64",
    "variant": "affine_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 1024.0,
    "seed": 43,
    "trainable": 114688,
    "total": 7615731200,
    "pct": 0.0015059355036060095,
    "eval_loss": 1.35,
    "train_runtime_s": 2457.0,
    "train_loss": 1.394
  },
  {
    "run_id": "qwen25_7b__seed43__single_q_l0_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 43,
    "trainable": 114688,
    "total": 7615731200,
    "pct": 0.0015059355036060095,
    "eval_loss": 1.554,
    "train_runtime_s": 2444.0,
    "train_loss": 1.587
  },
  {
    "run_id": "qwen25_7b__seed43__single_q_l14_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 43,
    "trainable": 114688,
    "total": 7615731200,
    "pct": 0.0015059355036060095,
    "eval_loss": 1.515,
    "train_runtime_s": 2410.0,
    "train_loss": 1.55
  },
  {
    "run_id": "qwen25_7b__seed43__single_q_l27_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 43,
    "trainable": 114688,
    "total": 7615731200,
    "pct": 0.0015059355036060095,
    "eval_loss": 1.619,
    "train_runtime_s": 2450.0,
    "train_loss": 1.646
  },
  {
    "run_id": "qwen25_7b__seed43__single_qkvo_l14_r4",
    "variant": "hidden_lora",
    "hidden_lora_rank": 4,
    "affine_alpha": 32.0,
    "seed": 43,
    "trainable": 90112,
    "total": 7615706624,
    "pct": 0.0011832388568648728,
    "eval_loss": 1.251,
    "train_runtime_s": 2424.0,
    "train_loss": 1.298
  },
  {
    "run_id": "qwen25_7b__seed43__vocab_only_r1",
    "variant": "hidden_lora",
    "hidden_lora_rank": 1,
    "affine_alpha": 32.0,
    "seed": 43,
    "trainable": 311296,
    "total": 7615927808,
    "pct": 0.004087433702732913,
    "eval_loss": 1.46,
    "train_runtime_s": 2452.0,
    "train_loss": 1.487
  },
  {
    "run_id": "qwen25_7b__seed43__vocab_only_r2",
    "variant": "hidden_lora",
    "hidden_lora_rank": 2,
    "affine_alpha": 32.0,
    "seed": 43,
    "trainable": 622592,
    "total": 7616239104,
    "pct": 0.008174533276837627,
    "eval_loss": 1.418,
    "train_runtime_s": 2460.0,
    "train_loss": 1.452
  },
  {
    "run_id": "qwen25_7b__seed44__affine_inlmh_s1_16",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 256.0,
    "seed": 44,
    "trainable": 232960,
    "total": 7615849472,
    "pct": 0.0030588839873541026,
    "eval_loss": 1.119,
    "train_runtime_s": 2459.0,
    "train_loss": 1.17
  },
  {
    "run_id": "qwen25_7b__seed44__affine_input_s1_64",
    "variant": "affine_input",
    "hidden_lora_rank": 16,
    "affine_alpha": 1024.0,
    "seed": 44,
    "trainable": 118272,
    "total": 7615734784,
    "pct": 0.0015529952572466053,
    "eval_loss": 1.228,
    "train_runtime_s": 2416.0,
    "train_loss": 1.279
  },
  {
    "run_id": "qwen25_7b__seed44__affine_lm_head_s1_64",
    "variant": "affine_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 1024.0,
    "seed": 44,
    "trainable": 114688,
    "total": 7615731200,
    "pct": 0.0015059355036060095,
    "eval_loss": 1.348,
    "train_runtime_s": 2462.0,
    "train_loss": 1.392
  },
  {
    "run_id": "qwen25_7b__seed44__single_q_l0_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 44,
    "trainable": 114688,
    "total": 7615731200,
    "pct": 0.0015059355036060095,
    "eval_loss": 1.55,
    "train_runtime_s": 2457.0,
    "train_loss": 1.584
  },
  {
    "run_id": "qwen25_7b__seed44__single_q_l14_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 44,
    "trainable": 114688,
    "total": 7615731200,
    "pct": 0.0015059355036060095,
    "eval_loss": 1.517,
    "train_runtime_s": 2443.0,
    "train_loss": 1.553
  },
  {
    "run_id": "qwen25_7b__seed44__single_q_l27_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 44,
    "trainable": 114688,
    "total": 7615731200,
    "pct": 0.0015059355036060095,
    "eval_loss": 1.617,
    "train_runtime_s": 2433.0,
    "train_loss": 1.645
  },
  {
    "run_id": "qwen25_7b__seed44__single_qkvo_l14_r4",
    "variant": "hidden_lora",
    "hidden_lora_rank": 4,
    "affine_alpha": 32.0,
    "seed": 44,
    "trainable": 90112,
    "total": 7615706624,
    "pct": 0.0011832388568648728,
    "eval_loss": 1.256,
    "train_runtime_s": 2430.0,
    "train_loss": 1.305
  },
  {
    "run_id": "qwen25_7b__seed44__vocab_only_r1",
    "variant": "hidden_lora",
    "hidden_lora_rank": 1,
    "affine_alpha": 32.0,
    "seed": 44,
    "trainable": 311296,
    "total": 7615927808,
    "pct": 0.004087433702732913,
    "eval_loss": 1.458,
    "train_runtime_s": 2444.0,
    "train_loss": 1.487
  },
  {
    "run_id": "qwen25_7b__seed44__vocab_only_r2",
    "variant": "hidden_lora",
    "hidden_lora_rank": 2,
    "affine_alpha": 32.0,
    "seed": 44,
    "trainable": 622592,
    "total": 7616239104,
    "pct": 0.008174533276837627,
    "eval_loss": 1.422,
    "train_runtime_s": 2440.0,
    "train_loss": 1.454
  },
  {
    "run_id": "qwen3_06b__seed42__affine_inlmh_s1_16",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 256.0,
    "seed": 42,
    "trainable": 66560,
    "total": 596116480,
    "pct": 0.011165603071399738,
    "eval_loss": 1.325,
    "train_runtime_s": 632.0,
    "train_loss": 1.363
  },
  {
    "run_id": "qwen3_06b__seed42__affine_input_s1_64",
    "variant": "affine_input",
    "hidden_lora_rank": 16,
    "affine_alpha": 1024.0,
    "seed": 42,
    "trainable": 33792,
    "total": 596083712,
    "pct": 0.005669002410184964,
    "eval_loss": 1.42,
    "train_runtime_s": 629.5,
    "train_loss": 1.457
  },
  {
    "run_id": "qwen3_06b__seed42__affine_lm_head_s1_64",
    "variant": "affine_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 1024.0,
    "seed": 42,
    "trainable": 32768,
    "total": 596082688,
    "pct": 0.005497223901929526,
    "eval_loss": 1.569,
    "train_runtime_s": 336.9,
    "train_loss": 1.602
  },
  {
    "run_id": "qwen3_06b__seed42__mergeable_tied_affine_rank32_s1_16",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 512.0,
    "seed": 42,
    "trainable": 66560,
    "total": 596116480,
    "pct": 0.011165603071399738,
    "eval_loss": 1.323,
    "train_runtime_s": 643.8,
    "train_loss": 1.365
  },
  {
    "run_id": "qwen3_06b__seed42__single_q_l0_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 49152,
    "total": 596099072,
    "pct": 0.008245609213094028,
    "eval_loss": 1.626,
    "train_runtime_s": 627.1,
    "train_loss": 1.658
  },
  {
    "run_id": "qwen3_06b__seed42__single_q_l14_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 49152,
    "total": 596099072,
    "pct": 0.008245609213094028,
    "eval_loss": 1.668,
    "train_runtime_s": 481.5,
    "train_loss": 1.704
  },
  {
    "run_id": "qwen3_06b__seed42__single_q_l27_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 49152,
    "total": 596099072,
    "pct": 0.008245609213094028,
    "eval_loss": 1.781,
    "train_runtime_s": 346.5,
    "train_loss": 1.802
  },
  {
    "run_id": "qwen3_06b__seed42__single_qkvo_l14_r4",
    "variant": "hidden_lora",
    "hidden_lora_rank": 4,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 40960,
    "total": 596090880,
    "pct": 0.006871435442864014,
    "eval_loss": 1.473,
    "train_runtime_s": 488.2,
    "train_loss": 1.522
  },
  {
    "run_id": "qwen3_06b__seed42__vocab_only_r1",
    "variant": "hidden_lora",
    "hidden_lora_rank": 1,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 305920,
    "total": 596355840,
    "pct": 0.051298231606149776,
    "eval_loss": 1.569,
    "train_runtime_s": 656.5,
    "train_loss": 1.598
  },
  {
    "run_id": "qwen3_06b__seed42__vocab_only_r2",
    "variant": "hidden_lora",
    "hidden_lora_rank": 2,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 611840,
    "total": 596661760,
    "pct": 0.1025438600254858,
    "eval_loss": 1.549,
    "train_runtime_s": 658.7,
    "train_loss": 1.581
  },
  {
    "run_id": "qwen3_06b__seed43__affine_inlmh_s1_16",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 256.0,
    "seed": 43,
    "trainable": 66560,
    "total": 596116480,
    "pct": 0.011165603071399738,
    "eval_loss": 1.325,
    "train_runtime_s": 629.9,
    "train_loss": 1.364
  },
  {
    "run_id": "qwen3_06b__seed43__affine_input_s1_64",
    "variant": "affine_input",
    "hidden_lora_rank": 16,
    "affine_alpha": 1024.0,
    "seed": 43,
    "trainable": 33792,
    "total": 596083712,
    "pct": 0.005669002410184964,
    "eval_loss": 1.418,
    "train_runtime_s": 628.4,
    "train_loss": 1.455
  },
  {
    "run_id": "qwen3_06b__seed43__affine_lm_head_s1_64",
    "variant": "affine_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 1024.0,
    "seed": 43,
    "trainable": 32768,
    "total": 596082688,
    "pct": 0.005497223901929526,
    "eval_loss": 1.568,
    "train_runtime_s": 335.7,
    "train_loss": 1.602
  },
  {
    "run_id": "qwen3_06b__seed43__mergeable_tied_affine_rank32_s1_16",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 512.0,
    "seed": 43,
    "trainable": 66560,
    "total": 596116480,
    "pct": 0.011165603071399738,
    "eval_loss": 1.323,
    "train_runtime_s": 640.6,
    "train_loss": 1.365
  },
  {
    "run_id": "qwen3_06b__seed43__single_q_l0_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 43,
    "trainable": 49152,
    "total": 596099072,
    "pct": 0.008245609213094028,
    "eval_loss": 1.629,
    "train_runtime_s": 623.7,
    "train_loss": 1.661
  },
  {
    "run_id": "qwen3_06b__seed43__single_q_l14_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 43,
    "trainable": 49152,
    "total": 596099072,
    "pct": 0.008245609213094028,
    "eval_loss": 1.667,
    "train_runtime_s": 480.1,
    "train_loss": 1.704
  },
  {
    "run_id": "qwen3_06b__seed43__single_q_l27_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 43,
    "trainable": 49152,
    "total": 596099072,
    "pct": 0.008245609213094028,
    "eval_loss": 1.779,
    "train_runtime_s": 342.5,
    "train_loss": 1.802
  },
  {
    "run_id": "qwen3_06b__seed43__single_qkvo_l14_r4",
    "variant": "hidden_lora",
    "hidden_lora_rank": 4,
    "affine_alpha": 32.0,
    "seed": 43,
    "trainable": 40960,
    "total": 596090880,
    "pct": 0.006871435442864014,
    "eval_loss": 1.477,
    "train_runtime_s": 485.8,
    "train_loss": 1.528
  },
  {
    "run_id": "qwen3_06b__seed43__vocab_only_r1",
    "variant": "hidden_lora",
    "hidden_lora_rank": 1,
    "affine_alpha": 32.0,
    "seed": 43,
    "trainable": 305920,
    "total": 596355840,
    "pct": 0.051298231606149776,
    "eval_loss": 1.57,
    "train_runtime_s": 654.6,
    "train_loss": 1.598
  },
  {
    "run_id": "qwen3_06b__seed43__vocab_only_r2",
    "variant": "hidden_lora",
    "hidden_lora_rank": 2,
    "affine_alpha": 32.0,
    "seed": 43,
    "trainable": 611840,
    "total": 596661760,
    "pct": 0.1025438600254858,
    "eval_loss": 1.549,
    "train_runtime_s": 656.5,
    "train_loss": 1.58
  },
  {
    "run_id": "qwen3_06b__seed44__affine_inlmh_s1_16",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 256.0,
    "seed": 44,
    "trainable": 66560,
    "total": 596116480,
    "pct": 0.011165603071399738,
    "eval_loss": 1.325,
    "train_runtime_s": 632.5,
    "train_loss": 1.364
  },
  {
    "run_id": "qwen3_06b__seed44__affine_input_s1_64",
    "variant": "affine_input",
    "hidden_lora_rank": 16,
    "affine_alpha": 1024.0,
    "seed": 44,
    "trainable": 33792,
    "total": 596083712,
    "pct": 0.005669002410184964,
    "eval_loss": 1.42,
    "train_runtime_s": 630.5,
    "train_loss": 1.457
  },
  {
    "run_id": "qwen3_06b__seed44__affine_lm_head_s1_64",
    "variant": "affine_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 1024.0,
    "seed": 44,
    "trainable": 32768,
    "total": 596082688,
    "pct": 0.005497223901929526,
    "eval_loss": 1.57,
    "train_runtime_s": 336.6,
    "train_loss": 1.602
  },
  {
    "run_id": "qwen3_06b__seed44__mergeable_tied_affine_rank32_s1_16",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 512.0,
    "seed": 44,
    "trainable": 66560,
    "total": 596116480,
    "pct": 0.011165603071399738,
    "eval_loss": 1.321,
    "train_runtime_s": 641.3,
    "train_loss": 1.365
  },
  {
    "run_id": "qwen3_06b__seed44__single_q_l0_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 44,
    "trainable": 49152,
    "total": 596099072,
    "pct": 0.008245609213094028,
    "eval_loss": 1.63,
    "train_runtime_s": 628.0,
    "train_loss": 1.663
  },
  {
    "run_id": "qwen3_06b__seed44__single_q_l14_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 44,
    "trainable": 49152,
    "total": 596099072,
    "pct": 0.008245609213094028,
    "eval_loss": 1.667,
    "train_runtime_s": 480.2,
    "train_loss": 1.703
  },
  {
    "run_id": "qwen3_06b__seed44__single_q_l27_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 44,
    "trainable": 49152,
    "total": 596099072,
    "pct": 0.008245609213094028,
    "eval_loss": 1.78,
    "train_runtime_s": 344.6,
    "train_loss": 1.802
  },
  {
    "run_id": "qwen3_06b__seed44__single_qkvo_l14_r4",
    "variant": "hidden_lora",
    "hidden_lora_rank": 4,
    "affine_alpha": 32.0,
    "seed": 44,
    "trainable": 40960,
    "total": 596090880,
    "pct": 0.006871435442864014,
    "eval_loss": 1.477,
    "train_runtime_s": 485.2,
    "train_loss": 1.527
  },
  {
    "run_id": "qwen3_06b__seed44__vocab_only_r1",
    "variant": "hidden_lora",
    "hidden_lora_rank": 1,
    "affine_alpha": 32.0,
    "seed": 44,
    "trainable": 305920,
    "total": 596355840,
    "pct": 0.051298231606149776,
    "eval_loss": 1.572,
    "train_runtime_s": 657.7,
    "train_loss": 1.601
  },
  {
    "run_id": "qwen3_06b__seed44__vocab_only_r2",
    "variant": "hidden_lora",
    "hidden_lora_rank": 2,
    "affine_alpha": 32.0,
    "seed": 44,
    "trainable": 611840,
    "total": 596661760,
    "pct": 0.1025438600254858,
    "eval_loss": 1.539,
    "train_runtime_s": 658.4,
    "train_loss": 1.572
  },
  {
    "run_id": "qwen3_17b__seed42__affine_inlmh_s1_16",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 256.0,
    "seed": 42,
    "trainable": 133120,
    "total": 1720708096,
    "pct": 0.007736349954385291,
    "eval_loss": 0.9116,
    "train_runtime_s": 888.6,
    "train_loss": 0.9618
  },
  {
    "run_id": "qwen3_17b__seed42__affine_input_s1_64",
    "variant": "affine_input",
    "hidden_lora_rank": 16,
    "affine_alpha": 1024.0,
    "seed": 42,
    "trainable": 67584,
    "total": 1720642560,
    "pct": 0.003927834959516519,
    "eval_loss": 0.9806,
    "train_runtime_s": 890.4,
    "train_loss": 1.031
  },
  {
    "run_id": "qwen3_17b__seed42__affine_lm_head_s1_64",
    "variant": "affine_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 1024.0,
    "seed": 42,
    "trainable": 65536,
    "total": 1720640512,
    "pct": 0.0038088141911655745,
    "eval_loss": 1.185,
    "train_runtime_s": 484.0,
    "train_loss": 1.225
  },
  {
    "run_id": "qwen3_17b__seed42__mergeable_tied_affine_rank32_s1_16",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 512.0,
    "seed": 42,
    "trainable": 133120,
    "total": 1720708096,
    "pct": 0.007736349954385291,
    "eval_loss": 0.9108,
    "train_runtime_s": 896.2,
    "train_loss": 0.9661
  },
  {
    "run_id": "qwen3_17b__seed42__single_q_l0_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 65536,
    "total": 1720640512,
    "pct": 0.0038088141911655745,
    "eval_loss": 1.273,
    "train_runtime_s": 879.5,
    "train_loss": 1.31
  },
  {
    "run_id": "qwen3_17b__seed42__single_q_l14_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 65536,
    "total": 1720640512,
    "pct": 0.0038088141911655745,
    "eval_loss": 1.273,
    "train_runtime_s": 683.9,
    "train_loss": 1.311
  },
  {
    "run_id": "qwen3_17b__seed42__single_q_l27_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 65536,
    "total": 1720640512,
    "pct": 0.0038088141911655745,
    "eval_loss": 1.395,
    "train_runtime_s": 493.3,
    "train_loss": 1.419
  },
  {
    "run_id": "qwen3_17b__seed42__single_qkvo_l14_r4",
    "variant": "hidden_lora",
    "hidden_lora_rank": 4,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 57344,
    "total": 1720632320,
    "pct": 0.0033327282844483586,
    "eval_loss": 1.058,
    "train_runtime_s": 692.6,
    "train_loss": 1.108
  },
  {
    "run_id": "qwen3_17b__seed42__vocab_only_r1",
    "variant": "hidden_lora",
    "hidden_lora_rank": 1,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 307968,
    "total": 1720882944,
    "pct": 0.017895929590897263,
    "eval_loss": 1.226,
    "train_runtime_s": 909.9,
    "train_loss": 1.256
  },
  {
    "run_id": "qwen3_17b__seed42__vocab_only_r2",
    "variant": "hidden_lora",
    "hidden_lora_rank": 2,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 615936,
    "total": 1721190912,
    "pct": 0.03578545504195644,
    "eval_loss": 1.199,
    "train_runtime_s": 908.6,
    "train_loss": 1.236
  },
  {
    "run_id": "qwen3_17b__seed43__affine_inlmh_s1_16",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 256.0,
    "seed": 43,
    "trainable": 133120,
    "total": 1720708096,
    "pct": 0.007736349954385291,
    "eval_loss": 0.9093,
    "train_runtime_s": 885.6,
    "train_loss": 0.9607
  },
  {
    "run_id": "qwen3_17b__seed43__affine_input_s1_64",
    "variant": "affine_input",
    "hidden_lora_rank": 16,
    "affine_alpha": 1024.0,
    "seed": 43,
    "trainable": 67584,
    "total": 1720642560,
    "pct": 0.003927834959516519,
    "eval_loss": 0.9788,
    "train_runtime_s": 885.1,
    "train_loss": 1.031
  },
  {
    "run_id": "qwen3_17b__seed43__affine_lm_head_s1_64",
    "variant": "affine_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 1024.0,
    "seed": 43,
    "trainable": 65536,
    "total": 1720640512,
    "pct": 0.0038088141911655745,
    "eval_loss": 1.183,
    "train_runtime_s": 481.6,
    "train_loss": 1.224
  },
  {
    "run_id": "qwen3_17b__seed43__mergeable_tied_affine_rank32_s1_16",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 512.0,
    "seed": 43,
    "trainable": 133120,
    "total": 1720708096,
    "pct": 0.007736349954385291,
    "eval_loss": 0.9108,
    "train_runtime_s": 899.4,
    "train_loss": 0.9659
  },
  {
    "run_id": "qwen3_17b__seed43__single_q_l0_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 43,
    "trainable": 65536,
    "total": 1720640512,
    "pct": 0.0038088141911655745,
    "eval_loss": 1.269,
    "train_runtime_s": 881.0,
    "train_loss": 1.31
  },
  {
    "run_id": "qwen3_17b__seed43__single_q_l14_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 43,
    "trainable": 65536,
    "total": 1720640512,
    "pct": 0.0038088141911655745,
    "eval_loss": 1.269,
    "train_runtime_s": 684.0,
    "train_loss": 1.309
  },
  {
    "run_id": "qwen3_17b__seed43__single_q_l27_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 43,
    "trainable": 65536,
    "total": 1720640512,
    "pct": 0.0038088141911655745,
    "eval_loss": 1.395,
    "train_runtime_s": 489.7,
    "train_loss": 1.419
  },
  {
    "run_id": "qwen3_17b__seed43__single_qkvo_l14_r4",
    "variant": "hidden_lora",
    "hidden_lora_rank": 4,
    "affine_alpha": 32.0,
    "seed": 43,
    "trainable": 57344,
    "total": 1720632320,
    "pct": 0.0033327282844483586,
    "eval_loss": 1.055,
    "train_runtime_s": 690.7,
    "train_loss": 1.107
  },
  {
    "run_id": "qwen3_17b__seed43__vocab_only_r1",
    "variant": "hidden_lora",
    "hidden_lora_rank": 1,
    "affine_alpha": 32.0,
    "seed": 43,
    "trainable": 307968,
    "total": 1720882944,
    "pct": 0.017895929590897263,
    "eval_loss": 1.232,
    "train_runtime_s": 908.9,
    "train_loss": 1.262
  },
  {
    "run_id": "qwen3_17b__seed43__vocab_only_r2",
    "variant": "hidden_lora",
    "hidden_lora_rank": 2,
    "affine_alpha": 32.0,
    "seed": 43,
    "trainable": 615936,
    "total": 1721190912,
    "pct": 0.03578545504195644,
    "eval_loss": 1.189,
    "train_runtime_s": 910.1,
    "train_loss": 1.227
  },
  {
    "run_id": "qwen3_17b__seed44__affine_inlmh_s1_16",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 256.0,
    "seed": 44,
    "trainable": 133120,
    "total": 1720708096,
    "pct": 0.007736349954385291,
    "eval_loss": 0.9136,
    "train_runtime_s": 892.0,
    "train_loss": 0.9631
  },
  {
    "run_id": "qwen3_17b__seed44__affine_input_s1_64",
    "variant": "affine_input",
    "hidden_lora_rank": 16,
    "affine_alpha": 1024.0,
    "seed": 44,
    "trainable": 67584,
    "total": 1720642560,
    "pct": 0.003927834959516519,
    "eval_loss": 0.9799,
    "train_runtime_s": 883.0,
    "train_loss": 1.028
  },
  {
    "run_id": "qwen3_17b__seed44__affine_lm_head_s1_64",
    "variant": "affine_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 1024.0,
    "seed": 44,
    "trainable": 65536,
    "total": 1720640512,
    "pct": 0.0038088141911655745,
    "eval_loss": 1.184,
    "train_runtime_s": 481.9,
    "train_loss": 1.225
  },
  {
    "run_id": "qwen3_17b__seed44__mergeable_tied_affine_rank32_s1_16",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 512.0,
    "seed": 44,
    "trainable": 133120,
    "total": 1720708096,
    "pct": 0.007736349954385291,
    "eval_loss": 0.9074,
    "train_runtime_s": 896.5,
    "train_loss": 0.963
  },
  {
    "run_id": "qwen3_17b__seed44__single_q_l0_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 44,
    "trainable": 65536,
    "total": 1720640512,
    "pct": 0.0038088141911655745,
    "eval_loss": 1.277,
    "train_runtime_s": 884.8,
    "train_loss": 1.313
  },
  {
    "run_id": "qwen3_17b__seed44__single_q_l14_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 44,
    "trainable": 65536,
    "total": 1720640512,
    "pct": 0.0038088141911655745,
    "eval_loss": 1.27,
    "train_runtime_s": 685.6,
    "train_loss": 1.309
  },
  {
    "run_id": "qwen3_17b__seed44__single_q_l27_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 44,
    "trainable": 65536,
    "total": 1720640512,
    "pct": 0.0038088141911655745,
    "eval_loss": 1.394,
    "train_runtime_s": 493.5,
    "train_loss": 1.418
  },
  {
    "run_id": "qwen3_17b__seed44__single_qkvo_l14_r4",
    "variant": "hidden_lora",
    "hidden_lora_rank": 4,
    "affine_alpha": 32.0,
    "seed": 44,
    "trainable": 57344,
    "total": 1720632320,
    "pct": 0.0033327282844483586,
    "eval_loss": 1.057,
    "train_runtime_s": 690.6,
    "train_loss": 1.11
  },
  {
    "run_id": "qwen3_17b__seed44__vocab_only_r1",
    "variant": "hidden_lora",
    "hidden_lora_rank": 1,
    "affine_alpha": 32.0,
    "seed": 44,
    "trainable": 307968,
    "total": 1720882944,
    "pct": 0.017895929590897263,
    "eval_loss": 1.205,
    "train_runtime_s": 909.6,
    "train_loss": 1.246
  },
  {
    "run_id": "qwen3_17b__seed44__vocab_only_r2",
    "variant": "hidden_lora",
    "hidden_lora_rank": 2,
    "affine_alpha": 32.0,
    "seed": 44,
    "trainable": 615936,
    "total": 1721190912,
    "pct": 0.03578545504195644,
    "eval_loss": 1.192,
    "train_runtime_s": 909.7,
    "train_loss": 1.228
  },
  {
    "run_id": "qwen3_8b__seed42__affine_inlmh_s1_16",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 256.0,
    "seed": 42,
    "trainable": 266240,
    "total": 8191001600,
    "pct": 0.00325039614202981,
    "eval_loss": 1.012,
    "train_runtime_s": 3079.0,
    "train_loss": 1.057
  },
  {
    "run_id": "qwen3_8b__seed42__affine_input_s1_64",
    "variant": "affine_input",
    "hidden_lora_rank": 16,
    "affine_alpha": 1024.0,
    "seed": 42,
    "trainable": 135168,
    "total": 8190870528,
    "pct": 0.001650227525120026,
    "eval_loss": 1.067,
    "train_runtime_s": 3097.0,
    "train_loss": 1.111
  },
  {
    "run_id": "qwen3_8b__seed42__affine_lm_head_s1_64",
    "variant": "affine_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 1024.0,
    "seed": 42,
    "trainable": 131072,
    "total": 8190866432,
    "pct": 0.001600221430640465,
    "eval_loss": 1.201,
    "train_runtime_s": 3069.0,
    "train_loss": 1.237
  },
  {
    "run_id": "qwen3_8b__seed42__single_q_l0_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 131072,
    "total": 8190866432,
    "pct": 0.001600221430640465,
    "eval_loss": 1.345,
    "train_runtime_s": 3054.0,
    "train_loss": 1.37
  },
  {
    "run_id": "qwen3_8b__seed42__single_q_l18_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 131072,
    "total": 8190866432,
    "pct": 0.001600221430640465,
    "eval_loss": 1.306,
    "train_runtime_s": 3099.0,
    "train_loss": 1.336
  },
  {
    "run_id": "qwen3_8b__seed42__single_q_l35_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 131072,
    "total": 8190866432,
    "pct": 0.001600221430640465,
    "eval_loss": 1.377,
    "train_runtime_s": 3057.0,
    "train_loss": 1.399
  },
  {
    "run_id": "qwen3_8b__seed42__single_qkvo_l18_r4",
    "variant": "hidden_lora",
    "hidden_lora_rank": 4,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 106496,
    "total": 8190841856,
    "pct": 0.0013001838134866317,
    "eval_loss": 1.124,
    "train_runtime_s": 3067.0,
    "train_loss": 1.16
  },
  {
    "run_id": "qwen3_8b__seed42__vocab_only_r1",
    "variant": "hidden_lora",
    "hidden_lora_rank": 1,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 312064,
    "total": 8191047424,
    "pct": 0.0038098180104005216,
    "eval_loss": 1.293,
    "train_runtime_s": 3110.0,
    "train_loss": 1.312
  },
  {
    "run_id": "qwen3_8b__seed42__vocab_only_r2",
    "variant": "hidden_lora",
    "hidden_lora_rank": 2,
    "affine_alpha": 32.0,
    "seed": 42,
    "trainable": 624128,
    "total": 8191359488,
    "pct": 0.007619345737594858,
    "eval_loss": 1.281,
    "train_runtime_s": 3094.0,
    "train_loss": 1.302
  },
  {
    "run_id": "qwen3_8b__seed43__affine_inlmh_s1_16",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 256.0,
    "seed": 43,
    "trainable": 266240,
    "total": 8191001600,
    "pct": 0.00325039614202981,
    "eval_loss": 1.01,
    "train_runtime_s": 3044.0,
    "train_loss": 1.055
  },
  {
    "run_id": "qwen3_8b__seed43__affine_input_s1_64",
    "variant": "affine_input",
    "hidden_lora_rank": 16,
    "affine_alpha": 1024.0,
    "seed": 43,
    "trainable": 135168,
    "total": 8190870528,
    "pct": 0.001650227525120026,
    "eval_loss": 1.069,
    "train_runtime_s": 3044.0,
    "train_loss": 1.115
  },
  {
    "run_id": "qwen3_8b__seed43__affine_lm_head_s1_64",
    "variant": "affine_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 1024.0,
    "seed": 43,
    "trainable": 131072,
    "total": 8190866432,
    "pct": 0.001600221430640465,
    "eval_loss": 1.202,
    "train_runtime_s": 3084.0,
    "train_loss": 1.239
  },
  {
    "run_id": "qwen3_8b__seed43__single_q_l0_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 43,
    "trainable": 131072,
    "total": 8190866432,
    "pct": 0.001600221430640465,
    "eval_loss": 1.356,
    "train_runtime_s": 3083.0,
    "train_loss": 1.377
  },
  {
    "run_id": "qwen3_8b__seed43__single_q_l18_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 43,
    "trainable": 131072,
    "total": 8190866432,
    "pct": 0.001600221430640465,
    "eval_loss": 1.308,
    "train_runtime_s": 3080.0,
    "train_loss": 1.338
  },
  {
    "run_id": "qwen3_8b__seed43__single_q_l35_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 43,
    "trainable": 131072,
    "total": 8190866432,
    "pct": 0.001600221430640465,
    "eval_loss": 1.38,
    "train_runtime_s": 3055.0,
    "train_loss": 1.401
  },
  {
    "run_id": "qwen3_8b__seed43__single_qkvo_l18_r4",
    "variant": "hidden_lora",
    "hidden_lora_rank": 4,
    "affine_alpha": 32.0,
    "seed": 43,
    "trainable": 106496,
    "total": 8190841856,
    "pct": 0.0013001838134866317,
    "eval_loss": 1.123,
    "train_runtime_s": 3072.0,
    "train_loss": 1.161
  },
  {
    "run_id": "qwen3_8b__seed43__vocab_only_r1",
    "variant": "hidden_lora",
    "hidden_lora_rank": 1,
    "affine_alpha": 32.0,
    "seed": 43,
    "trainable": 312064,
    "total": 8191047424,
    "pct": 0.0038098180104005216,
    "eval_loss": 1.287,
    "train_runtime_s": 3099.0,
    "train_loss": 1.307
  },
  {
    "run_id": "qwen3_8b__seed43__vocab_only_r2",
    "variant": "hidden_lora",
    "hidden_lora_rank": 2,
    "affine_alpha": 32.0,
    "seed": 43,
    "trainable": 624128,
    "total": 8191359488,
    "pct": 0.007619345737594858,
    "eval_loss": 1.276,
    "train_runtime_s": 3072.0,
    "train_loss": 1.297
  },
  {
    "run_id": "qwen3_8b__seed44__affine_inlmh_s1_16",
    "variant": "affine_input_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 256.0,
    "seed": 44,
    "trainable": 266240,
    "total": 8191001600,
    "pct": 0.00325039614202981,
    "eval_loss": 1.009,
    "train_runtime_s": 3090.0,
    "train_loss": 1.054
  },
  {
    "run_id": "qwen3_8b__seed44__affine_input_s1_64",
    "variant": "affine_input",
    "hidden_lora_rank": 16,
    "affine_alpha": 1024.0,
    "seed": 44,
    "trainable": 135168,
    "total": 8190870528,
    "pct": 0.001650227525120026,
    "eval_loss": 1.064,
    "train_runtime_s": 3095.0,
    "train_loss": 1.107
  },
  {
    "run_id": "qwen3_8b__seed44__affine_lm_head_s1_64",
    "variant": "affine_lm_head",
    "hidden_lora_rank": 16,
    "affine_alpha": 1024.0,
    "seed": 44,
    "trainable": 131072,
    "total": 8190866432,
    "pct": 0.001600221430640465,
    "eval_loss": 1.201,
    "train_runtime_s": 3078.0,
    "train_loss": 1.238
  },
  {
    "run_id": "qwen3_8b__seed44__single_q_l0_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 44,
    "trainable": 131072,
    "total": 8190866432,
    "pct": 0.001600221430640465,
    "eval_loss": 1.346,
    "train_runtime_s": 3076.0,
    "train_loss": 1.37
  },
  {
    "run_id": "qwen3_8b__seed44__single_q_l18_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 44,
    "trainable": 131072,
    "total": 8190866432,
    "pct": 0.001600221430640465,
    "eval_loss": 1.308,
    "train_runtime_s": 3062.0,
    "train_loss": 1.336
  },
  {
    "run_id": "qwen3_8b__seed44__single_q_l35_r16",
    "variant": "hidden_lora",
    "hidden_lora_rank": 16,
    "affine_alpha": 32.0,
    "seed": 44,
    "trainable": 131072,
    "total": 8190866432,
    "pct": 0.001600221430640465,
    "eval_loss": 1.378,
    "train_runtime_s": 3053.0,
    "train_loss": 1.399
  },
  {
    "run_id": "qwen3_8b__seed44__single_qkvo_l18_r4",
    "variant": "hidden_lora",
    "hidden_lora_rank": 4,
    "affine_alpha": 32.0,
    "seed": 44,
    "trainable": 106496,
    "total": 8190841856,
    "pct": 0.0013001838134866317,
    "eval_loss": 1.119,
    "train_runtime_s": 3064.0,
    "train_loss": 1.157
  },
  {
    "run_id": "qwen3_8b__seed44__vocab_only_r1",
    "variant": "hidden_lora",
    "hidden_lora_rank": 1,
    "affine_alpha": 32.0,
    "seed": 44,
    "trainable": 312064,
    "total": 8191047424,
    "pct": 0.0038098180104005216,
    "eval_loss": 1.289,
    "train_runtime_s": 3082.0,
    "train_loss": 1.308
  },
  {
    "run_id": "qwen3_8b__seed44__vocab_only_r2",
    "variant": "hidden_lora",
    "hidden_lora_rank": 2,
    "affine_alpha": 32.0,
    "seed": 44,
    "trainable": 624128,
    "total": 8191359488,
    "pct": 0.007619345737594858,
    "eval_loss": 1.277,
    "train_runtime_s": 3082.0,
    "train_loss": 1.299
  }
]
```
