# VisionModelLib  
A vision model library including ShapeModel, TextureModel, AppearanceModel, PatchRegressor and training tools for them.
  
This project is library only.  
Practical usage samples will be released in other repositories.  

---  

This project is written with java and rely on OpenCV 3.2.0 (already included in 'lib' folder)

In order to train models, you need to download MUCT Database.  
[http://www.milbo.org/muct/](http://www.milbo.org/muct/)  

Or you can use the models I pre-trained for you. (included in 'models' folder)  
The pre-trained models are face-shape-model, face-texture-model and an integrated face-appearance-model trained from MUCT database.  

You can also use other picture-landmark databases to train other models besides face model through revising 'MuctData.java'.

