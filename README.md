The files included in this repository are the result of months of work on my bachelor thesis. The title of my thesis is "SURE REACH Optimal Parameter Set and Model
Properties Investigation" and can be found within the files in this repository. Even though there is an already existing computer simulation of SURE_REACH made in Java I have decided to create an independent version solely based on the academic paper and its model description. The model is an interesting attempt at describing the way humans move and coordinate their movements on an example of a single planar 3-DOF arm.

Model Description

The model's main idea is for it to learn all by itself how to execute a sequence of motor commands to in order to approach a goal posture or point in space. In other words the model teaches itself through trial and error how to solve the inverse kinematics problem if the goal is a point in space and then for either types of goal it solves the motion planning problem in order to reach the set goal. The model is designed after the way humans perform motions hence the learning method utilized in the model is Hebbian learning which is a learning technique derived from neuroscience- "What fires together, wires together".

![image](https://user-images.githubusercontent.com/90681144/229618285-87a8e4ae-1917-4a06-a1ed-8bab3d2ad3ac.png)
![image](https://user-images.githubusercontent.com/90681144/229618315-777a10ec-3dfb-4bbf-a7da-2dbf06558c94.png)

Big thanks to the supervisors of this project; Fokko Jan Dijksterhuis and Frank van der Stappen as well as occasional guidence of Oliver Herbort (one of the authors of the SURE_REACH model)!!

