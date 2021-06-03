import os
import sys


# disable render output
def disable_output_start():
    logfile = "/dev/null"
    open(logfile, "a").close()
    old = os.dup(1)
    sys.stdout.flush()
    os.close(1)
    os.open(logfile, os.O_WRONLY)
    return old


def disable_output_end(old):
    os.close(1)
    os.dup(old)
    os.close(old)


def mkdir_safe(directory):
    try:
        os.makedirs(directory)
    except FileExistsError:
        pass
