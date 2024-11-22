from roboflow import Roboflow
rf = Roboflow(api_key="Zz5nOrbtGm8uqusXKTgY")
project = rf.workspace("byteboarder").project("waves-classification")
version = project.version(1)
dataset = version.download("yolov11")
                