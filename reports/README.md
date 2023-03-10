---
layout: default
nav_exclude: true
---

# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

where you instead should add your answers. Any other changes may have unwanted consequences when your report is auto
generated in the end of the course. For questions where you are asked to include images, start by adding the image to
the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

```markdown
![my_image](figures/<image>.<extension>)
```

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

will generate an `.html` page of your report. After deadline for answering this template, we will autoscrape
everything in this `reports` folder and then use this utility to generate an `.html` page that will be your serve
as your final handin.

Running

```bash
python report.py check
```

will check your answers in this template against the constrains listed for each question e.g. is your answer too
short, too long, have you included an image when asked to.

For both functions to work it is important that you do not rename anything. The script have two dependencies that can
be installed with `pip install click markdown`.

## Overall project checklist

The checklist is *exhaustic* which means that it includes everything that you could possible do on the project in
relation the curricilum in this course. Therefore, we do not expect at all that you have checked of all boxes at the
end of the project.

### Week 1

* [X] Create a git repository
* [X] Make sure that all team members have write access to the github repository
* [X] Create a dedicated environment for you project to keep track of your packages
* [X] Create the initial file structure using cookiecutter
* [X] Fill out the `make_dataset.py` file such that it downloads whatever data you need and
* [X] Add a model file and a training script and get that running
* [X] Remember to fill out the `requirements.txt` file with whatever dependencies that you are using
* [X] Remember to comply with good coding practices (`pep8`) while doing the project
* [ ] Do a bit of code typing and remember to document essential parts of your code
* [X] Setup version control for your data or part of your data
* [X] Construct one or multiple docker files for your code
* [X] Build the docker files locally and make sure they work as intended
* [X] Write one or multiple configurations files for your experiments
* [X] Used Hydra to load the configurations and manage your hyperparameters
* [ ] When you have something that works somewhat, remember at some point to to some profiling and see if
      you can optimize your code
* [X] Use Weights & Biases to log training progress and other important metrics/artifacts in your code. Additionally,
      consider running a hyperparameter optimization sweep.
* [X] Use Pytorch-lightning (if applicable) to reduce the amount of boilerplate in your code

### Week 2

* [X] Write unit tests related to the data part of your code
* [X] Write unit tests related to model construction and or model training
* [X] Calculate the coverage.
* [X] Get some continuous integration running on the github repository
* [X] Create a data storage in GCP Bucket for you data and preferable link this with your data version control setup
* [X] Create a trigger workflow for automatically building your docker images
* [X] Get your model training in GCP using either the Engine or Vertex AI
* [X] Create a FastAPI application that can do inference using your model
* [ ] If applicable, consider deploying the model locally using torchserve
* [X] Deploy your model in GCP using either Functions or Run as the backend

### Week 3

* [ ] Check how robust your model is towards data drifting
* [ ] Setup monitoring for the system telemetry of your deployed model
* [ ] Setup monitoring for the performance of your deployed model
* [ ] If applicable, play around with distributed data loading
* [ ] If applicable, play around with distributed model training
* [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed

### Additional

* [ ] Revisit your initial project description. Did the project turn out as you wanted?
* [ ] Make sure all group members have a understanding about all parts of the project
* [ ] Uploaded all your code to github

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:

Group 05

### Question 2
> **Enter the study number for each member in the group**
>
> Answer:

s221937, s222964, s222725, s210703

### Question 3
> **What framework did you choose to work with and did it help you complete the project?**
>
> Answer length: 100-200 words.
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.
>
> Answer:

We used [Transformers](https://github.com/huggingface/transformers) framework in our project. This framework implements state-of-the-art Machine Learning for Pytorch, and is a good fit for the goals of the project. From this framework we have used pre-trained the [BERT-TINY](https://huggingface.co/prajjwal1/bert-tiny), since it is a lighweight model and with some training in our dataset can achieve the spam classification goal with high accuracy.  We have started our project by using pre-trained model which help us to focus more implementing on the different techniques taught in the course. we implemented our BERT model to classifies sms text as spam or no spam using [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) where we have specified different parameters like batch size, epoch, optimizer, lr etc. for training our model.

## Coding environment

> In the following section we are interested in learning more about you local development environment.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development enviroment, one would have to run the following commands*
>
> Answer:

We used *pip* to install packages and *conda* to create the environment. The list of dependencies was auto-generated using *pipreqs* command which creates the *requirements.txt* file which contains a list of requirements based on imports in the project. However, we noticed that some dependecies were missing, so we added them manually. To get a complete copy of our development environment, the first step is to create a new conda environment (*conda create -n "my_environment python="version"*) and activate it (*conda activate "my_environment*). Our project environment was built with Python version 3.10. The next step is to clone the github project (*git clone https://github.com/lefteriskat/mlops_project.git*) and install everything from the requirements.txt file using this command:*pip install -r requirements.txt*.


### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. Did you fill out every folder or only a subset?**
>
> Answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
> Answer:

The overall structure of our project consists in cookiecutter template where we have filled out the src folder with the source code for data and model. Data folder contains the file which parses and splits the data into training, validation and test, while model folder contains the model, and scripts for training and prediction. When running the make_dataset.py script, the data from data/raw folder is saved as a csv file in the data/interim folder, splitted into three different files and saved into data/processed folder. After running the training script, the model is saved into the models folder and the parameters are saved into the lightning_logs. We have added tests folder, config folder which contains hydra configuration parameters for each script, and cloud_app folder that contains FastAPI function and the requirements needed to deploy the model in cloud. Moreover we added dvc files and dockefiles for the purpose of building docker image. 

### Question 6

> **Did you implement any rules for code quality and format? Additionally, explain with your own words why these**
> **concepts matters in larger projects.**
>
> Answer length: 50-100 words.
>
> Answer:

We have used flake8, isort and black in our project to ensure code quality and format. For flake8 we give max-line-length = 125 to allow for more explanetory variable name. Those tools are useful in helping to ensure that code is readable, maintainable, and free of errors. They can help to automate some of the tedious and error-prone tasks associated with code development, allowing developers to focus on the more important aspects of the larger project.


## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
>
> Answer length: 50-100 words.
>
> Example:
> *In total we have implemented X tests. Primarily we are testing ... and ... as these the most critical parts of our*
> *application but also ... .*
>
> Answer:

For this project, we have implemented two tests focusing on the dataset and model. For the dataset we tested that after splitting it into training, validation and testing datasets, the total number of inputs is equal to the total size of the dataset. Model testing ensures that output provided by the model has the desired shaped. Unfortunately, the training and prediction code is not cover by our tests.

### Question 8

> **What is the total code coverage (in percentage) of your code? If you code had an code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Answer length: 100-200 words.
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:

The total code coverage is 68% which includes tests for dataset and model code. The coverage rate for data script is 84% and for the model script is only 36%. The high percetange of data script was expected because the data script only splits the dataset and tokenizes it, so we managed to cover most functionalities. Since we did not test the training and prediction code at all and the model checks only one functionality, we expect that our code not to detect most of errors. However, beside code failures, there are many reasons why a project could fail such as wrong dependencies, wrong deployment of the code, not enough permission, corrupted data, installed different packages.

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:

We made use of both branches and pull requests in our project workflow. In our group, each member had at least one branch that each individual worked on. When we have done developing any specific feature we made a pull request where other group members checked it, comment and approved it. If there were any conflicts or errors found while reviewing the pull requests, the other members checked the code properly and find out the solution, correct it and merged the pull request. Working with pull requests and branches has the advantage that multiple developers can collaborate, work on the same files. In this way, the version control is mentained, the changes are reviewed to a codebase before they merged into the main branch.

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer:

We used dvc to pull data when building **docker** images. Firstly, we stored the data into the Google Drive as remote storage solution for our data, but this solution requires to authentic each time we try to either push or pull the data. To solve this problem, we used Google Cloud Storage to store the data into a public bucket which allows us to download files without being authenticated. The advantage of storing the data into the cloud is that the data is versioned for each experiment by replacing the large files into small metafile. Beside saving disk space on the local machine, the experiments also become reproducible in case the dataset changes. In the end, it helped us pull data from the cloud to control user permissions and consistency when building a docker image.

### Question 11

> **Discuss you continues integration setup. What kind of CI are you running (unittesting, linting, etc.)? Do you test**
> **multiple operating systems, python version etc. Do you make use of caching? Feel free to insert a link to one of**
> **your github actions workflow.**
>
> Answer length: 200-300 words.
>
> Example:
> *We have organized our CI into 3 separate files: one for doing ..., one for running ... testing and one for running*
> *... . In particular for our ..., we used ... .An example of a triggered workflow can be seen here: <weblink>*
>
> Answer:

We have organized our CI into 3 separate files: one for doing **unittesting**, one for running **isort** and one for running **flake8**. We used unittesting to test our code as it was stated before(the dataset and the model code). Isort is a python library which automatically separates the imports into sections and sorts them by type, so we used isort workflow to check if all imports are sorted correctly. The third workflow is represented by the flake8 which checked coding style and programming errors. For Isort and flake8 workflows we have used Python-version: 3.10.8 and the tests have run only on Ubuntu operating system while for unittesting we have used two Python-version: 3.10.8 and 3.8, and two operating systems: Ubuntu and Windows. Because workflows often reuse the same outputs or download dependencies from one run to another we used caching actions to make our workflows faster and more efficient. Caching actions create and restore a cache identified by a unique key. Overall, we made out of use of continous integration in the project because it tested our project automatically every time we pushed code into the main branch or we opened a pull request(PR) in the Github repository.
Here is our link to the unittesting actions workflow: `https://github.com/lefteriskat/mlops_project/actions/workflows/tests.yml`

## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: python my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:

We have used hydra to configure our project. It helps to load hyperparameters from a config/folder containing yaml files with hyperparameters for the model, predict and train in separate config files within subfolders. So when the config-files are filled in, the model gets all the necessary hyperparameters from them. For example, the model can be trained by just using the following commad: python src/models/train_model.py

### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:

As we stated in the previous question, we used hydra to keep track of the hyperparameters through config files. Each time the experiment was run, the respective hyperparameters were saved in the folder with the specific data and hour of the experiment. We ensured that our experiments were deterministic, by for example specifing the random seed in Pytroch and random state in the function train_test_split from the sklearn library. We further run an experiment with specified hyperparameters a couple of times and checked that the model weights and biases were the same, implying that we achieved reproducibility of experiments. To reproduce an experiment one would have to train the model with the hyperparameters specified in the coresponding config file.


### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:
We used the weights and biases service, which provieds tools to perform experiment tracking. As seen in the attached screenshot we have tracked the training loss and accuracy across epochs of training the model. Those metrics informed us about the ability of the model to learn patterns present in our data. However, evalution of the model on the training set is not sufficient when considering how well the model would behave on unseen data. To account for that, we have also tracked the validation loss and accuracy across the training process. Those metric informed us how well the model could generalise when deployed. Note, that in the attached screenshot only a couple of runs are present, since many runs were conducted to ensure that the model works as expected and no bugs were introcued. Those runs were left out as not informative for the final presentation. Further, seperate data set was put aside - the test set - and the model, after being trained for predefined number of epochs, was evaluated on it. Unfortunately, this metric is not included in weights and biases report, and one could add it to improve the user overall understanding of the model performance. As could be seen from the figures, the models achieved satisfactory performance after a couple of epochs of training, which is not suprising considering that the pretrained model was used.  
![this figure](figures/wb_project.png)

### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments? Include how you would run your docker images and include a link to one of your docker files.**
>
> Answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:

For our project we developed two docker images: one for training and one for deployment. The steps that we followed to use docker are: defined the dockerfiles, ran the dockerfile to build a docker image, ran the docker image to create a docker container. Every time when some code is merged to the main branch, an action is triggered in Cloud Build which builds a new training docker image in the Container Registry. Running the training docker image can be done locally by pulling the image (*docker pull gcr.io/<project-id>/<image_name>:<image_tag>*) on the local machine and then creating the docker container or running in the cloud. To run locally a docker image use this command: `docker run --name experiment1 trainer:latest`. Link to the docker image: <https://console.cloud.google.com/gcr/images/dtumlops-374515/global/project_train?project=dtumlops-374515>

### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:

Debugging methods varied slightly based on each group member, however we came to the conclusion that in this project the most effective debugging methods were printing different variables, especially when it comes to shapes of tensors, and googling encuntered errors. We used the latter method frequently when working in the services provided by the google cloud. Altough, we did not use a python debugger, we agree that it could be beneficial to incorporate it into our coding practice. Regrettably, we did not profile our code. Doing so would be high on our priority list if we decided to further improve our project.

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:

We used the following services:
* Cloud Storage Buckets: Buckets in GCP are basic containers where you can store data in the cloud.
* Cloud Build: Cloud Build is a service that executes your builds on Google Cloud.
* Vertex AI: Vertex AI is a machine learning (ML) platform that lets you train and deploy ML models and AI applications.
* Container Registry: Container Registry is a service for storing private container images.
* Cloud Functions: Google Cloud Functions is a serverless execution environment.
* Cloud Run: Cloud Run is a managed compute platform that enables you to run containers that are invocable via requests or events. Cloud Run is serverless: it abstracts away all infrastructure management, so you can focus on what matters most, building great applications.

### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Answer length: 100-200 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:

We used the compute engine when the Cloud Build Trigger was enabled to build the docker image. We used the standard virtual machine (VM) because that is the machine that Cloud Build runs the default builds on. The VM used is E2 machine with 2vCPU and 4 GB memory. Since building docker images is usually disk space intensive and time consuming, virtual machines have helped us to use large scale hardware and run processes in the background. Furthermore, in this way we ensured that the model could be trained without any further inference. Compute engine is an useful tool because provides security, backups, scalability and storage efficency.

### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

![Gcp buckets](figures/gcp_bucket.png)
![Trained model bucket content](figures/trained_model_bucket.png)

### Question 20

> **Upload one image of your GCP container registry, such that we can see the different images that you have stored.**
> **You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

![GCP Container Registry](figures/cloud_container_registry.png)

### Question 21

> **Upload one image of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

![GCP Build History](figures/build_history.png)


### Question 22

> **Did you manage to deploy your model, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:

We used Cloud Build Service to create a trigger which is activated every time we push to the main branch of our github repository. When activated a docker image is created and stored in the Container Registry from our training dockerfile.  When the image is created from the previous step we manually create a custom job which runs a container using the generated image which runs the training of our model and also extracts the trained model via torchscript. Then the extracted model is pushed to a gcp bucket. After we created a fast_api application which loads our model from the gcp bucket that was extracted from the custom training job and after getting an sms text as input it provides the prediction of our model regarding if it is spam or not. Afterwards we created a docker image containing this application which we manually pushed to the google cloud container. Then we used cloud run to deploy this image and make it available for users.
 You can invoke our app by running the following command:
 *`curl -X 'GET'   'https://gcp-cloud-app-2-avta7xclua-lz.a.run.app/check_if_spam?sms_text=You%20won%203000%20dollars'   -H 'accept: application/json'`*

### Question 23

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

We did not manage to implement monitoring. Nevertheless monitoring is a very significant part of a real-world model deployment since 
it measures the model's performance and inform us about different events that maybe require some actions from us. 
For example, errors, where we have to provide a fix asap to make our deployed model available again. 
Another case where we have to take an action, and it is ML related, is when the accuracy of our model has dropped significantly due to data drifting
and we need to re-train it with new data. Therefore, monitoring is a must have for real-world ML applications where the model is used for several puproses and you have to ensure both quality for our predictions and availability.

### Question 24

> **How many credits did you end up using during the project and what service was most expensive?**
>
> Answer length: 25-100 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ...*
>
> Answer:

On average we spent 20$ each. The most expensive service was the storage which was the only one continuously consuming 
credits during both the exercises and te project whereas the other services were consuming just when we were using them.

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 25

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally in your own words, explain the**
> **overall steps in figure.**
>
> Answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and puch to github, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:

The overall architecture of our system can be seen in ![this figure](figures/architecture.png). The starting point of the diagram is our local machine, where we computed the data, created the model classes, integrated the **wandb** service and used **hydra** to get configuration parameters. After parsing the data, we ensured the control of it by uploading it to cloud data storage from where we pull it every time a docker image is created. The diagram shows that whenever we **commit** and **push** code to **github**, it auto triggers the github actions (**unittesting, flake8 and isort**) and a **cloud build** starts building a **docker** image with the latest version of the code and dataset. Once the image is built, it can be found in the **Container Registry**. At this point, an user should create a custom job on **Vertex AI** to run the latest docker image from which results the **trained model** exported to the **Bucket storage**. Looking at the **deployment** phase, we created a **FastAPI** function which loads the model from the bucket storage and creates a prediction for a given input. Using this function, we created a docker container to deploy it in the **Cloud Run** because docker always can use the dependencies of our application. We also used **Cloud Functions** to deploy our model.


### Question 26

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:

The biggest challenges we encountered in the project were related to deploying the FastAPI application locally and building the docker image in the cloud. The reason we struggled to build docker images in the cloud was due to lack of permissions and the fact that building a docker image took around half an hour. Even though we took the advantage of using dvc to extract the data, so we could download the data without being logged in, the problem was caused by the wandb service. In the end, we managed to solve this problem by using the Secret Manager service to generate a secret key needed to connect to the wandb service. Our second biggest challenge was the local deployment which hit us again with a lack of permissions. We wanted to download the model from Bucket Storage and make a new prediction based on it, but the docker container could not connect to our cloud storage. Unfortunately, this problem was not solved by our team and we deployed the model directly in the cloud. Another problem our team encountered was the lack of disk space, as the project generally required more local memory than we expected. This was especially problematic to some of the group members who were not always able to build docker. The general struggle and sometimes the source of frustration was the fact that building docker images required a considerable amount of time and minor changes could be implemented only after waiting for quite a while. It significantly slowed our overall progress when we were still learning how to properly set docker files.

### Question 27

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project**
>
> Answer length: 50-200 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
>
> Answer:

Eleftherios (s222725) was in charge of developing initial model, setting up github actions, writing the unittests, incorporating continous integration, setting up triggers in the cloud, and final deployment of the model in the cloud. Tymoteusz (s221937) was in charge of initial incorporating the Pytroch-Lightning framework and refactorization of the code, including the Hydra config files, including the weights and biases service, deployment of the model via the function cloud. Denisa (s222964) was in charge of parsing the data, incorporating the data version control, developing the model container for training the model, developing the fastAPI application and creating a docker containers for launching it in the cloud. Shariful (s210703) was in charge of setting up the cookie cutter project structure, setting up the hyperparameters for the model, model evaluation, creating the docker containers. All members contributed to the discussion of the project structure, creation of the pipeline, and writing the final report. To all other tasks not specifically mentioned above, all studetns contribued equally.
