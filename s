[33mb786d6d3[m[33m ([m[1;36mHEAD[m[33m -> [m[1;32mmain[m[33m, [m[1;31morigin/main[m[33m, [m[1;31morigin/HEAD[m[33m)[m MAPE de attentions pool
[33m4c3d1777[m Novo metodo de pooling
[33m736440c6[m Mudanca no metodo de pooling e correcao de bugs
[33m0a539064[m Novo algoritimo para analise e reducao de norm
[33m242d9e69[m Adicionado análise do modelo encoder
[33m0afd736a[m[33m ([m[1;32mteste-antigo[m[33m)[m added new results models
[33m7d05cab1[m update files
[33mcc6dfccf[m add UniTS x128 pretrain checkpoint
[33m452f04b7[m added new models
[33m20272804[m all experiments completed
[33m49a2b6f5[m fix bugs
[33md77d675c[m update scripts
[33m943b6cc1[m added new datasets
[33ma25a09b2[m update
[33m4fcd5fe1[m added new datasets
[33mcf7b12e3[m fixed file size limit
[33m15dd59b8[m fixed file size limit
[33mf36f2f8c[m added new dataset M5
[33ma0cd7efb[m Merge branch 'main' of https://github.com/AlexCohenDambros/Master-FoundationModels
[33m55da660d[m added results from the models trained on the benchmark
[33m66acf357[m fix bug
[33m061e9908[m feat: add training script for initial model on benchmark datasets
[33m1809bc28[m added new models
[33m2118d07a[m added filtered datasets
[33m91853876[m Adjusted code for model training
[33m3d3ef8a1[m fix Bayesian search method
[33m4efcdc5f[m fix bugs
[33m67aa031a[m added new dataset
[33mc7ab5e88[m added new test
[33m08520796[m drop datasets
[33m44200646[m fix script benchmark
[33mec06b11c[m added time series encoder
[33m377937f3[m Added new datasets for experimentation
[33mb899b0b0[m Added Bayesian training and parameter search.
[33mcd1f1c58[m config paper
[33me4f86cc1[m added new dataset
[33m916d5961[m add new results exp
[33m9058869e[m new update
[33ma55151d2[m new updates
[33mb7031e4d[m new updates
[33mec32d207[m new results
[33m8ebf39f2[m Added new results analysis.
[33ma2743dd0[m new update test
[33m9a0b3a5c[m new update test
[33m913e0457[m Added results analysis
[33m432a21a1[m Added analysis of parameter tests for SP
[33m77c4abcb[m router routing collapse correction applied
[33m55bc6cce[m feat(training): add feature to train model with multiple parameter configurations
[33meda23945[m fix bug
[33m55ab4fd3[m new update
[33m837f5f9c[m added new feature
[33m39996172[m fix bug
[33m10d20872[m update models
[33m4ff063e7[m fix training error
[33me3c69db6[m added new results
[33mfbc17173[m added custom loss function
[33m7b63fdf9[m new update
[33m281ee016[m added results top_k = 1 using minmax
[33maa1ed001[m fixing mymoe
[33m489794a2[m adding new results
[33mbc378f0e[m fix bug
[33m0588f0cd[m added Parallel
[33m52e3d1c3[m fix bug
[33m609f35d8[m feat(analysis): add train loss analysis for experiments
[33ma65cb5af[m feat: add GPU support
[33m1400e382[m feat(analysis): add new Bayesian test analyses
[33m08df5c1a[m fix(preprocessing): correctly add MinMax normalization
[33m346df29c[m feat(router): add support for top_k = 1
[33ma00d163a[m feat(experiments): add new model experiments with no significant improvements
[33mfdb93602[m fix(results): fix error in all_results
[33md0b7bcdb[m Remove large training log files from repository
[33mfa92f335[m feat(training): train new models for the proposed method
[33m62c8164c[m feat: add support for parallelizing code across cores
[33m13ba33bf[m change cuda
[33mb0e04ea0[m fix bug
[33m2b1e0c1c[m set model to device
[33m539ece71[m update experts
[33md1429aa0[m fix error
[33ma98ff070[m fix error cuda
[33mea3e78f1[m fix bug
[33m70b9d44b[m feat: add GPU support
[33m3b84069c[m feat: add GPU support
[33meb347359[m fix(training): fix bug preventing training of new models
[33m59462c78[m chore(experts): update experts file paths
[33me08082b9[m feat(router): add new models to router training
[33m6d0911b1[m feat(evaluation): add new experimental results and include Bayesian test
[33m4552c6d7[m update training_all_models
[33ma300e640[m fix error main
[33m46c8e69a[m fix(training): fix errors in model training
[33m75e2485e[m feat(router): add initial tests for noisy top-k with balancing loss
[33m1898a84b[m chore: organize project files
[33m2b5c5304[m feat(router): initialize weights uniformly and add per-model normalization (MinMax, StandardScaler)
[33mbaedd9b7[m feat(evaluation): add average processing time for models
[33md33b10b1[m fix torch.load
[33m6d1ec447[m fix ranking
[33mce931cef[m feat(evaluation): add model ranking and prediction time calculation in seconds
[33mdff31582[m feat(experiments): finalize and compute experimental results
[33m1a2f01f3[m feat(evaluation): fix model testing code and add plot_all_results.ipynb for yearly and product analysis
[33m74ee9913[m feat(training): add training for all models over 5-year period
[33m2c57075c[m feat(experts): add new foundation models as experts
[33m43ebe591[m fix bug trust_remote_code
[33m68c382e3[m fix bug trust_remote_code
[33mc710edf1[m chore: add requirements file
[33m0507ca4f[m fix(training): fix issues in model training
[33m6dd0dd65[m chore: apply clean code improvements
[33m6db58519[m feat(training): add code to train multiple models on 5-year global experimental dataset
[33m1535dbc1[m feat(training): add code to train multiple models across different states
[33m68f64d19[m feat(router): add log to print selected experts during training
[33mbbd76f08[m feat(test): add averaging of foundation models in model evaluations
[33m8c4f8867[m feat(training): add MinMax and StandardScaler, best results with Standard
[33md8933ec7[m fix(predict): update predict_from_model to support 2D tensor input
[33m2ce7f6d7[m chore: remove unnecessary folders
[33m8aba2e5f[m feat(EarlyStopping): add EarlyStopping to model training, fix bugs and errors
[33m05304ce5[m fix: bug fixes and adds new experts to MoE
[33m4d2b37db[m feat(horizon): the model can now forecast any forecast horizon
[33m5b852a94[m fix(example): temporarily removing the load-balancing regularizer (simple)
[33mb02d79e1[m new update MoE_V2
[33m30a00e7c[m new update MoE_v2
[33m9d329864[m update MoE_v2
[33m051e7e6b[m new moe_v2
[33mcc1e2751[m update code and new issue found on the experts train
[33m7e77a4b4[m new updates
[33mefd9d3c8[m new updates
[33m3c3d4359[m update files
[33m617e84c6[m first commit
