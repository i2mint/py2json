Tools for json serialization of python objects

# py2json

Here we tackle the problem of serializing a python object into a json. 

Json is a convenient choice for web request responses or working with mongoDB for instance. 

It is usually understood that we serialize an object to be able to deserialize it to recover the original object: Implicit in this is some definition of equality, which is not as trivial as it may seem. Usually **some** aspects of the deserialized object will be different, so we need to be clear on what should be the same.

For example, we probably don't care if the address of the deserialized object is different. But we probably care that it's key attributes are the same.

What should guide us in deciding what aspects of an object should be recovered? 

Behavior. 

The only value of an object is behavior that will ensue. This may be the behavior of all or some of the methods of a serialized instance, or the behavior of some other functions that will depend on the deserialized object. 

Our approach to converting a python object to a json will touch on some i2i cornerstones that are more general: Conversion and contextualization. 


# Behavior equivalence: What do we need an object to have?

Say we are given the code below.

```python
def func(obj):
    return obj.a + obj.b

class A:
    e = 2
    def __init__(self, a=0, b=0, c=1, d=10):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        
    def target_func(self, x=3):
        t = func(self)
        tt = self.other_method(t)
        return x * tt / self.e
    
    def other_method(self, x=1):
        return self.c * x
```

Which we use to make the following object
```python
obj = A(a=2, b=3)
```


Say we want to json-serialize this so that a deserialized object `dobj` is such that for all valid `obj`, resulting `dobj`, and valid `x` input:

```
obj.target_func(x) == A.target_func(obj, x) == A.target_func(dobj, x)
```
The first equality is just a reminder of a python equivalence. 
The second equality is really what we're after. 

When this is true, we'll say that `obj` and `dobj` are equivalent on `A.target_func` -- or just "equivalent" when the function(s) it should be equivalent is clear. 

To satisfy this equality we need `dobj` to:
- Contain all the attributes it needs to be able to compute the `A.target_func` function -- which means all the expressions contained in that function or, recursively, any functions it calls. 
- Such that the values of a same attribute of `obj` and `dobj` are equivalent (over the functions in the call try of the target function that involve these attributes.

Let's have a manual look at it. 
First, you need to compute `func(self)`, which will require the attributes `a` and `b`. 
Secondly, you'll meed to computer `other_method`, which uses attribute `c`. 
Finally, the last expression, `x * tt / self.e` uses the attribute `e`. 

So what we need to make sure we serialize the attributes: `{'a', 'b', 'c', 'e'}`. 

That wasn't too hard. But it could get convoluted. Either way, we really should use computers for such boring tasks!

That's something `py2json` would like to help you with.
