MiniMind embedding/head 预训练实验打包

文件说明：
1. 实验.md
   完整实验说明（变体、配置、结果解读）

2. minimind_original_three_seed_summary.csv
   MiniMind 原始预训练数据，S1-S13，三 seed (42/123/2026) 汇总

3. fineedu_gpt2_three_seed_summary.csv
   FineWeb-Edu/GPT-2 数据，S1-S13，三 seed 汇总

来源目录：/home/wz/projects/mypro/im_exp/minimind/

4. fineedu-tied-vs-untied-grad-mean.svg
   FineEdu/GPT-2 预训练 S1 vs S2 梯度范数曲线（seed 42/123/2026 平均），矢量图

5. scripts/plot_fineedu_tied_vs_untied_grad.py
   重绘脚本；原始数据在 logs/fineedu-gpt2-6b-seed{42,123,2026}/s1|s2_grad_stats.jsonl
