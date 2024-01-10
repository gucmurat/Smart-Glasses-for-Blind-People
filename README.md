# Smart-Glasses-for-Blind-People

### One-Minute Presentation

![image](https://github.com/gucmurat/Smart-Glasses-for-Blind-People/assets/79659434/8a0c4160-1fdb-46b3-87df-4e52d711b0a2)

### Contract

- Even in the year of 2023, visually impaired individuals face many challenges because of the fact that the cities, places, and public transportations are not designed by considering the disabled people. Despite the fact that there are several facilitators such as tactile pavings, audible pedestrian signals, guide dogs etc. , there can be several financial, accessibility and availability issues and these issues cause the life quality of blind people is visibly low in society. 
- As a senior design project, we offer smart glasses that allow the wearer to continue their life somewhat normally without seeing the surroundings. At the end of the semester, our goal is to design glasses which scans the environment and recognize the objects, which are critical for the user of the glasses, and inform the user about the objects themselves and their distance from the blind person if necessary. 


### Build

- to create conda environment
  - conda create --name senior-project python=3.9
- activate
  - conda activate senior-project
- deactivate
  - conda deactivate senior-project
- install packages
  - cd ./scripts
- running server
  - cd ./scripts/server
  - pip install -r requirements
  - python server.py
- running client
  - cd ./scripts/client
  - pip install -r requirements
  - (terminal 1) python client.py
  - (terminal 2) uvicorn speech:app --port 8002
