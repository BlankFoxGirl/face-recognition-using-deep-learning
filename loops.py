class Break(Exception): pass
class Break2(Exception): pass
class Break3(Exception): pass
class Break4(Exception): pass

def loop(CALLBACK, EXCEPT=None):
    if EXCEPT is None:
        EXCEPT = [Break]

    try:
        CALLBACK()
    except EXCEPT:
        pass