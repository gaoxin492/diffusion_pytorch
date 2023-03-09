from inspect import isfunction

def exists(x):
    return x is not None

# 有val时返回val，val为None时返回d
def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d