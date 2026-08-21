"""
3. Eval the trained context model from #2 after shuffling proejcted embeddings (without retraining)

The performance should drop signifcantly (otherwise model might have learnt a shortcut that doesn't even depedn on sensor readings - which is bad)
If the shortcut was present, then the F1 score of the #2 woul look artifically high

Basically a check to see whether the identity of the sensor window is driving the prediction or not 
if it is, then the performace scores shoudl be significnatly differnt. 

"""