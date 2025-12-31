SFFR: Spatial–Frequency Feature Reconstruction for Multispectral Aerial Object Detection
====
Abstract
-------
Recent multispectral object detection methods have primarily focused on spatial-domain feature fusion based on CNNs or Transformers, while the potential of frequency-domain feature remains underexplored. 
In this work, we propose a novel Spatial and Frequency Feature Reconstruction method (SFFR) method, which leverages the spatial-frequency feature representation mechanisms of the Kolmogorov–Arnold Network (KAN) to reconstruct complementary representations in both spatial and frequency domains prior to feature fusion.
The core components of SFFR are the proposed Frequency Component Exchange KAN (FCEKAN) module and Multi-Scale Gaussian KAN (MSGKAN) module. 
The FCEKAN introduces an innovative selective frequency component exchange strategy that effectively enhances the complementarity and consistency of cross-modal features based on the frequency feature of RGB and IR images.
The MSGKAN module demonstrates excellent nonlinear feature modeling capability in the spatial domain. By leveraging multi-scale Gaussian basis functions, it effectively captures the feature variations caused by scale changes at different UAV flight altitudes, significantly enhancing the model’s adaptability and robustness to scale variations.
It is experimentally validated that our proposed FCEKAN and MSGKAN modules are complementary and can effectively capture the frequency and spatial semantic features respectively for better feature fusion.
Extensive experiments on the SeaDroneSee, DroneVehicle and DVTOD datasets demonstrate the superior performance and significant advantages of the proposed method in UAV multispectral  object perception task.

Paper download in [SFFR](https://ieeexplore.ieee.org/document/11240218)

Overview
-----
![](https://github.com/qchenyu1027/SFFR/blob/master/data/images/overview.png)

Installation
-----

```
https://github.com/qchenyu1027/SFFR.git
pip install -r requirements.txt
```

Weights
----
**DVTOD**

Link: https://pan.baidu.com/s/12iWrfK7rnFvw9IhpaZ8n3Q?pwd=9869 (9869)

**SeaDroneSee**

Link: https://pan.baidu.com/s/1Q_btR2GmH86V8kXz13fD2A?pwd=g7j2 (g7j2)

**DroneVehicle**

Link: https://pan.baidu.com/s/1Z-IAu8fIbovD12oW1mBd7A?pwd=9dn4 (9dn4)

Cite us
---
@ARTICLE{11240218,

  author={Zuo, Xin and Qu, Chenyu and Zhan, Haibo and Shen, Jifeng and Yang, Wankou},
  
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  
  title={SFFR: Spatial–Frequency Feature Reconstruction for Multispectral Aerial Object Detection}, 
  
  year={2025},
  
  volume={63},
  
  number={},
  
  pages={1-11},
  
  doi={10.1109/TGRS.2025.3631708}}


