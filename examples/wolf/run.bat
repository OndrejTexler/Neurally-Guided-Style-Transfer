@echo off

:: Patch-Based synthesis will run on resolution not higher than 1Mpx (the result will be then upsampled to 6000x4000)
..\..\styletransfer.exe --style style.jpg --neural_result neuralGatys.jpg --out_path result_patch_based_max_mp_1.jpg --patch_based_max_mp 1

:: Patch-Based synthesis will run on resolution not higher than 4Mpx (the result will be then upsampled to 6000x4000)
..\..\styletransfer.exe --style style.jpg --neural_result neuralGatys.jpg --out_path result_patch_based_max_mp_4.jpg --patch_based_max_mp 4

:: Patch-Based synthesis will run on resolution not higher than 4Mpx (the result will be then upsampled to 6000x4000)
:: Also, forces the patch-based synthesis to run on CPU even if this version is compiled with GPU support
:: This might be handy in case you set patch_based_max_mp to higher number, and the patch-based synthesis requires more GPU memory than you have
..\..\styletransfer.exe --style style.jpg --neural_result neuralGatys.jpg --out_path result_patch_based_max_mp_4_cpu.jpg --patch_based_max_mp 4 --patch_based_backend "CPU"

:: Patch-Based synthesis will run on resolution not higher than 1Mpx (the result will be then upsampled to 6000x4000)
:: Also, the result will resemble target photograph's colors.
:: This might or might not lead to pleasing results - just try it :-)
..\..\styletransfer.exe --style style.jpg --neural_result neuralGatys.jpg --out_path result_patch_based_max_mp_1_recolor_by_target.jpg --patch_based_max_mp 1 --target target.jpg --recolor_by_target

:: Patch-Based synthesis will run on resolution not higher than 1Mpx (the result will be then upsampled to 6000x4000)
:: Also, target photograph will be used to guide the synthesis and the original geometry should be more preserved; however, some style attributes might be restrained. 
:: This might or might not lead to better results - just try it :-)
..\..\styletransfer.exe --style style.jpg --neural_result neuralGatys.jpg --out_path result_patch_based_max_mp_1_guide_by_target.jpg --patch_based_max_mp 1 --target target.jpg --guide_by_target

pause