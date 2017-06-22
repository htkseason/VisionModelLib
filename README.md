# VisionModelLib  
This project implements ShapeModel, TextureModel, AppearanceModel, PatchModel and training tools for them.  
  
---  
  
### Instructions  
This project is __LIBRARY ONLY__. Examples are included in other repositories.  
This project is written with java and rely on [OpenCV 3.2.0](http://opencv.org/releases.html) java interface  

In order to train models, you need to download [Muct database](http://www.milbo.org/muct/) or other similar databases OR you can use the models I pre-trained for you. (included in 'models' folder)  
The pre-trained models are face-shape-model, face-texture-model, an integrated face-appearance-model and 76-points-patch-model which are all trained from Muct database.  
  
---
  
### Examples  
  
__Example 1 : AAM-Fitting (Using Appearance-Model, Texture-Model and Shape-Model)__  
[github.com/htkseason/AAM-Fitting](https://github.com/htkseason/AAM-Fitting)  
<img src="https://github.com/htkseason/AAM-Fitting/blob/master/demo.jpg" width="60%" alt="AAM-Fitting" />  
  
__Example 2 : CLM-Tracking (Using Patch-Model and Shape-Model)__  
[github.com/htkseason/CLM-Tracking](https://github.com/htkseason/ASM-Tracking)  
<img src="https://github.com/htkseason/CLM-Tracking/blob/master/demo.jpg" width="60%" alt="CLM-Tracking" />  
  
__Example 3 : Face-Twisting (Using Shape-Model and triangle mesh)__  
[github.com/htkseason/TwistFace](https://github.com/htkseason/TwistFace)  
<img src="https://github.com/htkseason/TwistFace/blob/master/demo.jpg" width="60%" alt="Face-Twisting" />  
  
---  
  
### References  
[Cootes and C.J.Taylor. Statistical Models of Appearance for Computer Vision. University of Manchester, March 2004](http://www.face-rec.org/algorithms/AAM/app_models.pdf)  
  
[github.com/MasteringOpenCV](https://github.com/MasteringOpenCV/code)  