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
  
__Example 1 : AAM-Fitting__  
[github.com/htkseason/AAM-Fitting](https://github.com/htkseason/AAM-Fitting)  
<img src="https://github.com/htkseason/AAM-Fitting/blob/master/demos/demo2.jpg" width="75%"/>  
  
__Example 2 : CLM-Tracking / SDM__  
[github.com/htkseason/CLM-Tracking](https://github.com/htkseason/ASM-Tracking)  
<img src="https://github.com/htkseason/CLM-Tracking/blob/master/demo.jpg" width="60%" alt="CLM-Tracking" /> 
<img src="https://github.com/htkseason/CLM-Tracking/blob/master/demo-patch-visualization.png" width="75%" /> 
  
__Example 3 : Apply to Android -- Face-Sticker-Camera__  
[github.com/htkseason/VmlFacial](https://github.com/htkseason/VmlFacial)  
<img src="https://github.com/htkseason/VmlFacial/blob/master/demos/demo1.jpg" width="35%"/>  
  
__Example 4 : Archery Target Score__
[github.com/htkseason/Archery-Target-Score](https://github.com/htkseason/Archery-Target-Score)  
<img src="https://github.com/htkseason/Archery-Target-Score/blob/master/demo.jpg" width="75%"/>  
  
  
---  
  
### References  
[Cootes and C.J.Taylor. Statistical Models of Appearance for Computer Vision. University of Manchester, March 2004](http://www.face-rec.org/algorithms/AAM/app_models.pdf)  
[github.com/MasteringOpenCV](https://github.com/MasteringOpenCV/code)  