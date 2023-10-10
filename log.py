import sys, datetime
DEBUG=False
id=None

def setId(newId):
    global id
    id=newId

def write(m, ANSI="", DUMP=None):
    global id
    print(ANSI + "[{}.{}]{}\033[0m".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), id, m))
    printDump(DUMP, ANSI=ANSI)
    sys.stdout.flush()

def printDump(dump = None, ANSI=""):
    if dump is None:
        return

    if ANSI != "":
        print(ANSI)

    print(dump)

    if ANSI != "":
        print("\033[0m") # Reset output.

def condenseAppend(append):
    resp = ""
    if len(append) == 0:
        return resp
    for item in append:
        resp += "[{}]".format(item)
    return resp

def debug(m, APPEND=[], DUMP=None):
    if DEBUG == True:
        ansi="\033[1;37;40m"
        write("[DEBUG]{} {}".format(condenseAppend(APPEND), m), ANSI=ansi, DUMP=DUMP)

def info (m, APPEND=[], DUMP=None):
    write("[INFO]{} {}".format(condenseAppend(APPEND), m))
    printDump(DUMP)

def success (m, APPEND=[], DUMP=None):
    color="\033[1;32;40m"
    write("[SUCCESS]{} {}".format(condenseAppend(APPEND), m), ANSI=color, DUMP=DUMP)

def error(m, APPEND=[], DUMP=None):
    color = "\033[1;31;40m"
    write("[ERROR]{} {}".format(condenseAppend(APPEND), m), ANSI=color, DUMP=DUMP)

def warning(m, APPEND=[], DUMP=None):
    color = "\033[1;33;40m"
    write("[WARNING]{} {}".format(condenseAppend(APPEND), m), ANSI=color, DUMP=DUMP)

def critical(m, APPEND=[], DUMP=None):
    ansi = "\033[5m\033[1;35;40m"
    write("[CRITICAL]{} {}".format(condenseAppend(APPEND), m), ANSI=ansi, DUMP=DUMP)

def exception(m, APPEND=[], DUMP=None):
    color = "\033[0;30;47m"
    write("[EXCEPTION]{} {}".format(condenseAppend(APPEND), m), ANSI=color, DUMP=DUMP)

def log(m, APPEND=[], DUMP=None):
    write("[LOG]{}{}".format(condenseAppend(APPEND), m), DUMP=DUMP)