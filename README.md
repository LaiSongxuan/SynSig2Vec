##### The SynSig2Vec method [1,2] for dynamic signature synthesis and verification. 

This repository implements a sigma-lognormal-based signature synthesis algorithm and a novel model for learning signature representations. The sigma-lognormal parameter extraction algorithm is based on Reilly et al.'s method [3]. To use this repository, you should first download the DeepSignDB database [4]. If you find this repository useful, please cite our papers [1,2].

##### Pipeline

1. Signature preprocessing. Please refer to the "preprocess" directory. Preprocessed data will be saved in the "data" directory.

2. Sigma-lognormal parameter extraction. Refer to the "sigma_lornormal" directory. The extractParams_full.py script considers the pen-ups and pen-downs as a single component, whereas extractParams_stroke.py only considers the pen-downs. Extracted parameters will be saved in the "sigma_lognormal/params" directory.

3. Signature synthesis and model training. Refer to the "Sig2Vec" subdirectory.  

   1) main.sh: Training Sig2Vec models

   2) evaluate.sh: Extraction of signature feature vectors using trained models.

   3) verifier_*.py: Verification of the signatures according to the protocol of DeepSignDB [3].

   When running the main.py script for the first time, it takes a long time to generate synthetic signatures, which will be cached in the "Sig2Vec/cache" directory for reuse.

##### Environment 

Tested with PyTorch 1.6 and Python 3.7. 

##### References

[1] Lai S, Jin L, Zhu Y, et al. SynSig2Vec: Forgery-free learning of dynamic signature representations by Sigma Lognormal-based Synthesis and 1D CNN[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2021.

[2] Lai S, Jin L, Lin L, et al. SynSig2Vec: Learning representations from synthetic dynamic signatures for real-world verification[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2020, 34(01): 735-742.

[3] O’Reilly C, Plamondon R. Development of a Sigma–Lognormal representation for on-line signatures[J]. Pattern recognition, 2009, 42(12): 3324-3337.

[4] Tolosana R, Vera-Rodriguez R, Fierrez J, et al. DeepSign: Deep on-line signature verification[J]. IEEE Transactions on Biometrics, Behavior, and Identity Science, 2021.
