# Litter Finding In the Wild

## Inspiration
Habitat loss, overhunting, and non-native species are the three main causes of species endangerment. We are worried that using computer vision to detect endangered animals may lead to an even easier hunt and find for those already endangered animals. Rather, we decided to use computer vision models to detect and potentially raise awareness of the cause of endangerment.
We focused on trash in the wild, as it not only leads to habitat damage but trash-eating and trapping are also among the deadliest harm to wildlife. 
__Challenge__: Identifying trash in the wild is considered harder than identifying it in a rather cleaner indoor ground, as wild environments are much more complex. 

## Pipeline Structure
After a photo is fed into the pipeline, it will be sent to a YOLOv9 model to first identify any presence of trash. Then the parts of the photo that are identified to be trash will be sent into a ResNet-18 model to further classify the type of trash, which potentially can identify which pieces of trash are more dangerous and urgently need to be cleaned up. Then it's our job as humans to go clean up :) OR we can combine this with another automatic trash-cleaning machine! 

## Future 
We are proposing a system that can potentially be developed in the future where drones are sent out to patrol the wild, having a camera that observes the ground. The pictures observed will go through the proposed pipeline utilizing YOLO and ResNet to identify then classify the trash. Notification will be sent back to the system for the trash found, with a high priority for large clumps or dangerous trash (such as draggling plastics and batteries). 

## Dataset
We used the UAVVaste dataset to fine-tune the pre-trained YOLOv9 for drawing bounding boxes around the trash, and the TACO dataset to fine-tune the pre-trained ResNet model for classifying trash. 
