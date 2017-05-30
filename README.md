# VisionModelLib  
This project includes implement of ShapeModel, TextureModel, AppearanceModel, PatchRegressor and training tools for them.  
This project is __LIBRARY ONLY__. Practical usage samples are included in other repositories.  
  
---  
  
## Instructions  
This project is written with java and rely on [OpenCV 3.2.0](http://opencv.org/releases.html) java interface  

In order to train models, you need to download [Muct database](http://www.milbo.org/muct/) or other similar databases OR you can use the models I pre-trained for you. (included in 'models' folder)  
The pre-trained models are face-shape-model, face-texture-model, an integrated face-appearance-model and 76-points-patch-model which are all trained from MUCT database.  
  
---
  
## Examples  
  
__Example 1 : AAM-Fitting (Usage of Appearance-Model, Texture-Model and Shape-Model)__  
[github.com/htkseason/AAM-Fitting](https://github.com/htkseason/AAM-Fitting)  
![AAM-Fitting](https://github.com/htkseason/AAM-Fitting/blob/master/demo.jpg)  
  
__Example 2 : ASM-Tracking (Usage of Patch-Regressor and Shape-Model)__  
[github.com/htkseason/ASM-Tracking](https://github.com/htkseason/ASM-Tracking)  
![ASM-Tracking](https://github.com/htkseason/ASM-Tracking/blob/master/demo.jpg)  
  
__Example 3 : Face-Twisting (Usage of Shape-Model and triangle mesh)__  
[github.com/htkseason/TwistFace](https://github.com/htkseason/TwistFace)  
![Face-Twisting](https://github.com/htkseason/TwistFace/blob/master/demo.jpg)  
  
---  
  
## References  
[Cootes and C.J.Taylor. Statistical Models of Appearance for Computer Vision. University of Manchester, March 2004](http://www.face-rec.org/algorithms/AAM/app_models.pdf)  
  
[github.com/MasteringOpenCV](https://github.com/MasteringOpenCV/code)  