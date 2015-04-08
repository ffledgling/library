def objective_function(accuracy=None, balance=None, overlap=None, margin=None, purity=None):
    # Take all the different things we measure and create a single unifying function
    # use the value of this function to decide if something a good choice
    # Values returned should always be b/w 0 and 1.

    value = 1.0

    if accuracy is not None:
        value *= accuracy
    if balance is not None:
        value *= balance
    if overlap:
        value *= (1 - (sum((1 for x in filter(lambda x: x>=0.05, (overlap.values()))))/len(overlap)))
        #value *= 1.0
    if margin:
        value *= margin
    if purity is not None:
        value *= purity

    return value
