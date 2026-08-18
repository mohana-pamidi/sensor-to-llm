# sensor-to-llm

Need to tokenize the prompt but not the sensor data. 

sensor data (128, 9) --> CNN Encoder (dim = 128) For understandign the sensor signal --> 
projector (dim --> hidden dim size = 960 ) ( tralating the understadning itself into an llm token instead of tokening the data itself into text)

encoder -> projector -> frozen LLM -> Linear(->6) 

inspired by LLaVA: Large Language and vision Assitant 
