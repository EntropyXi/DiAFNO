# DiAFNO
# Integrating Fourier Neural Operator with Diffusion Model for Autoregressive Predictions of Three-dimensional Turbulence

Code accompanying the manuscript titled ["Integrating Fourier Neural Operator with Diffusion Model for Autoregressive Predictions of Three-dimensional Turbulence"](https://arxiv.org/abs/2512.12628), authored by Yuchi Jiang, Yunpeng Wang, Huiyu Yang and Jianchun Wang.

## Abstract

Accurately autoregressive prediction of three-dimensional (3D) turbulence has been one of the most challenging problems for machine learning approaches. Diffusion models have demonstrated high accuracy in predicting two-dimensional (2D) turbulence, but their applications in 3D turbulence are relatively limited. To achieve reliable autoregressive predictions of 3D turbulence, we propose the DiAFNO model which integrates the implicit adaptive Fourier neural operator (IAFNO) with diffusion model. IAFNO can effectively capture the global frequency and structural features, which is crucial for global consistent reconstructions of the denoising process in diffusion models. Furthermore, based on conditional generation from diffusion models, we design an autoregressive framework in DiAFNO to achieve long-term stable predictions of 3D turbulence. The proposed DiAFNO model is systematically trained and tested separately with fixed hyperparameters in several types of 3D turbulence, including forced homogeneous isotropic turbulence (HIT) at Taylor Reynolds number $Re_{\lambda}\approx100$, decaying HIT at initial Taylor Reynolds number at $Re_{\lambda}\approx100$ and turbulent channel flow at friction Reynolds numbers $Re_{\tau}\approx395$ and $Re_{\tau}\approx590$ with case-specific training at each Reynolds number. The results in the \textit{a posteriori} tests demonstrate that DiAFNO exhibits a significantly higher prediction accuracy in most of the analyzed statistics (such as the velocity spectra, the root-mean-square (RMS) values of both velocity and vorticity, and Reynolds stresses), as compared to the elucidated diffusion model (EDM) and the traditional large-eddy simulation (LES) using dynamic Smagorinsky model (DSM). Although DiAFNO is not optimal in certain statistics, its overall performance is substantially better than all baseline models (EDM and DSM). Meanwhile, we record the time taken by machine learning models and DSM during the inference stage. Ignoring training costs, the well-trained DiAFNO achieves higher inference efficiency than EDM and LES with DSM.

## Dataset

The datasets can be downloaded at [IAFNO_fDNS_kaggle](https://www.kaggle.com/datasets/yuchirichardjiang/coarsened-fdns-data-iafno).

## Citation

arXiv version:
```
@misc{jiang2026integratingfourierneuraloperator,
      title={Integrating Fourier Neural Operator with Diffusion Model for Autoregressive Predictions of Three-dimensional Turbulence}, 
      author={Yuchi Jiang and Yunpeng Wang and Huiyu Yang and Jianchun Wang},
      year={2026},
      eprint={2512.12628},
      archivePrefix={arXiv},
      primaryClass={physics.flu-dyn},
      url={https://arxiv.org/abs/2512.12628}, 
}
```

This manuscript has been accepted by Acta Mechanica Sinica with citing inform: Acta Mech. Sin. 43, 360674 (2027), DOI: 10.1007/s10409-026-60674-x. When the final version of the article provided by the journal becomes retrievable, please cite it using the information of the final version. Many thanks.
