def objective_function_multiplicative(accuracy=None, balance=None, overlap=None, margin=None, purity=None):
    # Take all the different things we measure and create a single unifying function
    # use the value of this function to decide if something a good choice
    # Values returned should always be b/w 0 and 1.

    value = 1.0

    if accuracy is not None:
        value *= accuracy
    if balance is not None:
        value *= balance
    if overlap:
        #value *= (1 - (sum((1 for x in filter(lambda x: x>=0.05, (overlap.values()))))/len(overlap)))
        value *= (1 - (sum((1 for x in overlap.values() if x >= 0.05 ))/len(overlap)))
        #value *= 1.0
    if margin:
        value *= margin
    if purity is not None:
        value *= purity

    return value

def objective_function_wieghted_sum(accuracy=None, balance=None, overlap=None, margin=None, purity=None):
    # Values returned and passed to this function should always be b/w 0 and 1.
    # Compute weighted sums

    # Parameters:
    A = 1.0
    B = 1.0
    C = 1.0
    D = 1.0
    E = 1.0

    value = 0.0

    if accuracy is not None:
        value += (A*accuracy)
    if balance is not None:
        value += (B*balance)
    if overlap:
        value += (C*(1 - (sum((1 for x in filter(lambda x: x>=0.05, (overlap.values()))))/len(overlap))))
    if margin:
    # Not normalized yet, keep out
        pass
    if purity is not None:
        value += (E*purity)

    return value

def objective_function(accuracy=None, balance=None, overlap=None, margin=None, purity=None):
    # Wrapper function to call the correct underlying objective function based on need and implementation

    # Take all the different things we measure and create a single unifying function
    # use the value of this function to decide if something a good choice
    # Values returned should always be b/w 0 and 1.

    return objective_function_multiplicative(accuracy, balance, overlap, margin, purity)
