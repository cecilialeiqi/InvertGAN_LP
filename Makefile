relu:
		python test_noisy_sparse.py ReLU 3 1 ./network/trained_vae_20_60_784_nobias.pth 0 20,60,784 invert linf
		python test_noisy_sparse.py ReLU 3 1 ./network/trained_vae_20_60_784_nobias.pth 1 20,60,784 invert relax
		python test_noisy_sparse.py ReLU 3 0 flaceholder 0 20,100,500 invert linf
		python test_noisy_sparse.py ReLU 3 0 flaceholder 1 20,100,500 invert relax

leakyrelu:
		python test_noisy_sparse.py LeakyReLU 4 1 ./network/trained_vae_leakyrelu_20_200_500_784.pth 0 20,200,500,784 invert linf
		python test_noisy_sparse.py LeakyReLU 4 1 ./network/trained_vae_leakyrelu_20_200_500_784.pth 1 20,200,500,784 invert relax
		python test_noisy_sparse.py LeakyReLU 4 0 flaceholder 0 20,100,200,500 invert linf
		python test_noisy_sparse.py LeakyReLU 4 0 flaceholder 1 20,100,200,500 invert relax
		
choose_k:
		python test_noisy_sparse.py ReLU 3 0 flaceholder 0 20,250,600 choose_k linf
		python test_noisy_sparse.py LeakyReLU 3 0 flaceholder 0 20,250,600 choose_k linf 

sensing:
		python test_noisy_sparse.py ReLU 3 1 ./network/trained_vae_20_60_784_nobias.pth 0 20,60,784 sensing linf
