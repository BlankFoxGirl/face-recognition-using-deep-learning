import sys, datetime
DEBUG=False

def write(m):
    print("[{}]{}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), m))
    sys.stdout.flush()

def debug(m):
    if DEBUG == True:
        write("[DEBUG]{}".format(m))

def info (m):
    write("[INFO]{}".format(m))

def error(m):
    write("[ERROR]{}".format(m))

def warning(m):
    write("[WARNING]{}".format(m))

def critical(m):
    write("[CRITICAL]{}".format(m))

def exception(m):
    write("[EXCEPTION]{}".format(m))

def log(m):
    write("[LOG]{}".format(m))