import math
from functools import reduce

def sigmoid(input):
    """
    sigmoid
    :param input:
    :return:
    """
    return 1/(1+math.exp(-input))


def softmax(inputs):
    """
    softmax
    :param inputs:  list
    :return:
    """
    outputs = [math.exp(input) for input in inputs]
    output_sum = reduce(lambda x, y: x + y, outputs)
    outputs = [output/output_sum for output in outputs]
    return outputs



if __name__ == '__main__':
    input_test = [2., 3.]
    input_test2 = 3.0
    print(softmax(input_test))
    print(sigmoid(input_test2))
