# Reconstructing-Dynamic-Soft-Tissue-with-Stereo-Endoscope-Based-on-a-Single-layer-Network
Three-dimensional reconstruction of dynamic soft-tissue scenes with stereo endoscopic images is very challenging. A simple self-supervised stereo reconstruction framework is proposed, which bridges the traditional geometric deformable models and the newly revived neural networks. The equivalence between the classical Thin Plate Spline (TPS) model and a single-layer fully-connected or convolutional network is studied. By alternate training of two TPS equivalent networks within the self-supervised framework, disparity priors are learnt from the past stereo frames of target tissues to obtain an optimized disparity basis, on which the disparity maps of subsequent frames can be estimated more accurately without reducing computational efficiency and robustness. Finally, the proposed method is verified based on stereo-endoscopic videos recorded by the da Vinci surgical robots.
Key words Soft-tissue 3D reconstruction, disparity estimation, neural networks, thin-plate spline, surgical robot


The dataset can be found in https://imperialcollegelondon.app.box.com/s/kits2r3uha3fn7zkoyuiikjm1gjnyle3

If you want to use the trained dataset, run "fullZ_real_expect_perimg", you may need to change the dataset path in the code.
For people who want to train their models.
You need to run "createA" to create an A matrix for TPS.
And then run the "fullZ_real_train.ipynb"
Run "fullZ_real_expect_perimg" to expect the image depth.
Finally, use "show_3D_face" to generate the 3D surface of the object.
