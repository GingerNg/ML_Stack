

def reduce_mean(inputs):
    inputs = inputs(lambda x, y: x + y, inputs)
    if isinstance(inputs,list):
        reduce_mean(inputs)