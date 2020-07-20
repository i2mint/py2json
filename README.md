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
