# FineEdu S13/S5 梯度关系摘要

说明：MiniMind S13/S5 当时 `GRAD_LOG_INTERVAL=0`，没有 `*_grad_stats.jsonl`，这里是 FineEdu/GPT-2 三 seed 的梯度统计。

## S13：输入词表加性 LoRA vs 输入 hidden 乘性 LoRA

- seed 42: vocab LoRA mean norm 0.011519, mul LoRA mean norm 0.019523, mul/vocab 1.69x, vocab/shared 14.67%, mul/shared 24.86%, embed/unemb cosine mean -0.0020, unemb/embed 1.90x
- seed 123: vocab LoRA mean norm 0.011737, mul LoRA mean norm 0.013963, mul/vocab 1.19x, vocab/shared 15.17%, mul/shared 18.05%, embed/unemb cosine mean -0.0014, unemb/embed 1.94x
- seed 2026: vocab LoRA mean norm 0.011942, mul LoRA mean norm 0.015246, mul/vocab 1.28x, vocab/shared 15.29%, mul/shared 19.52%, embed/unemb cosine mean -0.0005, unemb/embed 1.91x

## S5：输出侧 LoRA

- seed 42: output LoRA mean norm 0.023013, output/shared 29.31%, output_lora_a 0.018931, output_lora_b 0.012506, embed/unemb cosine mean -0.0016, unemb/embed 1.93x
- seed 123: output LoRA mean norm 0.027557, output/shared 34.92%, output_lora_a 0.024475, output_lora_b 0.012263, embed/unemb cosine mean -0.0003, unemb/embed 1.90x
- seed 2026: output LoRA mean norm 0.023584, output/shared 30.33%, output_lora_a 0.019766, output_lora_b 0.012605, embed/unemb cosine mean -0.0006, unemb/embed 1.95x

## 与 S1/S4/S6/S12 的 tied split 对比

- seed 42: s1: cos -0.0009, unemb/embed 1.93x; s4: cos -0.0007, unemb/embed 1.97x; s5: cos -0.0016, unemb/embed 1.93x; s6: cos -0.0006, unemb/embed 1.87x; s12: cos 0.0062, unemb/embed 1.93x; s13: cos -0.0020, unemb/embed 1.90x
- seed 123: s1: cos -0.0005, unemb/embed 1.93x; s4: cos 0.0007, unemb/embed 1.95x; s5: cos -0.0003, unemb/embed 1.90x; s6: cos 0.0016, unemb/embed 1.85x; s12: cos -0.0022, unemb/embed 1.92x; s13: cos -0.0014, unemb/embed 1.94x
- seed 2026: s1: cos -0.0018, unemb/embed 1.96x; s4: cos -0.0008, unemb/embed 1.98x; s5: cos -0.0006, unemb/embed 1.95x; s6: cos 0.0008, unemb/embed 1.85x; s12: cos -0.0030, unemb/embed 1.95x; s13: cos -0.0005, unemb/embed 1.91x

## 与效果的对应

### s13

- seed 42: delta_vs_s1_loss -0.005260, mul/vocab 1.69x, cos -0.0020
- seed 123: delta_vs_s1_loss -0.008652, mul/vocab 1.19x, cos -0.0014
- seed 2026: delta_vs_s1_loss -0.000524, mul/vocab 1.28x, cos -0.0005
### s5

- seed 42: delta_vs_s1_loss -0.001614, output/shared 29.31%, cos -0.0016
- seed 123: delta_vs_s1_loss -0.002853, output/shared 34.92%, cos -0.0003
- seed 2026: delta_vs_s1_loss +0.001764, output/shared 30.33%, cos -0.0006
