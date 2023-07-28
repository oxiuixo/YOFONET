# YOFONET
Official code implementation of 2023 ChinaMM paper "Omnidirectional Image Quality Assessment based on Feature Sharing and Adaptive Viewport Fusion"
# Abstract
In the process of capturing, transmitting and storing omnidirectional images, a credible quality assessment approach is crucial to transmitting system optimization and improvement of quality of experience. Viewport-based methods is a promising direction for omnidirectional image quality assessment, which takes full advantage of viewing characteristics of users. However, the selection and processing of viewport images are complex, and the DNN based feature extraction should be conducted on each of the viewport. To solve the aforementioned issues, we propose an omnidirectional image quality assessment framework based on feature sharing and adaptive fusion of multiple viewport features, which transforms the viewport segmentation from the pixel domain to the shared feature domain. The method uses a single backbone to extract both semantic features and multi-scale quality features from ERP projection images at high resolution scales, and then splits the features based on Fibonacci uniform sampling. Finally, the framework aggregates the local quality features guided by semantic features. The experimental results on CVIQ and OIQA validate the accuracy of the proposed framework and demonstrates the effectiveness of each module.
# Overall structure of the proposed YOFONet
![image](https://github.com/oxiuixo/YOFONET/assets/58387120/1f864f29-a223-43fc-af9e-b5878db63720)
