from subprocess import check_output


def print_loaded_lmod_modules():
    lmod = os.environ["LMOD_CMD"]
    print(f"lmod={lmod}")
    exec(check_output([lmod, "python", "list"]))


def init_lmod_modules(module_names):
    lmod = os.environ["LMOD_CMD"]
    print(f"lmod={lmod}")

    for module_name in module_names:
        exec(check_output([lmod, 'python', 'load', module_name]))
    exec(check_output([lmod, "python", "list"]))


# import init_lmod_modules
# init_lmod_modules.init_lmod_modules(["gcc/8.4.0-cuda", "cuda/10.1"])
# init_lmod_modules.print_loaded_lmod_modules()

import subprocess
import tempfile
import os


def ShellEval(command_str):
    """
    Evaluate the supplied command string in the system shell.
    Operates like the shell eval command:
       -  Environment variable changes are pulled into the Python environment
       -  Changes in working directory remain in effect
    """
    temp_stdout = tempfile.SpooledTemporaryFile()
    temp_stderr = tempfile.SpooledTemporaryFile()
    # in broader use this string insertion into the shell command should be given more security consideration
    subprocess.call("""trap 'printf "\\0`pwd`\\0" 1>&2; env -0 1>&2' exit; %s""" % (command_str,), stdout=temp_stdout,
                    stderr=temp_stderr, shell=True)
    temp_stdout.seek(0)
    temp_stderr.seek(0)
    all_err_output = temp_stderr.read()
    allByteStrings = all_err_output.split(b'\x00')
    command_error_output = allByteStrings[0]
    new_working_dir_str = allByteStrings[1].decode(
        'utf-8')  # some risk in assuming index 1. What if commands sent a null char to the output?

    variables_to_ignore = ['SHLVL', 'COLUMNS', 'LINES', 'OPENSSL_NO_DEFAULT_ZLIB', '_']

    newdict = dict([tuple(bs.decode('utf-8').split('=', 1)) for bs in allByteStrings[2:-1]])
    for (varname, varvalue) in newdict.items():
        if varname not in variables_to_ignore:
            if varname not in os.environ:
                # print("New Variable: %s=%s"%(varname,varvalue))
                os.environ[varname] = varvalue
            elif os.environ[varname] != varvalue:
                # print("Updated Variable: %s=%s"%(varname,varvalue))
                os.environ[varname] = varvalue
    deletedVars = []
    for oldvarname in os.environ.keys():
        if oldvarname not in newdict.keys():
            deletedVars.append(oldvarname)
    for oldvarname in deletedVars:
        # print("Deleted environment Variable: %s"%(oldvarname,))
        del os.environ[oldvarname]

    if os.getcwd() != os.path.normpath(new_working_dir_str):
        # print("Working directory changed to %s"%(os.path.normpath(new_working_dir_str),))
        os.chdir(new_working_dir_str)
    # Display output of user's command_str.  Standard output and error streams are not interleaved.
    print(temp_stdout.read().decode('utf-8'))
    print(command_error_output.decode('utf-8'))


# print(f"Path before: {os.environ['PATH']}")
# ShellEval("module purge; module load gcc/8.4.0 cuda/10.1")
# print(f"Path after: {os.environ['PATH']}")

keys = set()

def prepend_path(path, value, delim=":"):
    keys.add(path)
    if not len(value):
        print(f"Empty value for path {path}")
        return

    if path in os.environ and len(os.environ[path]) and os.environ[path] != delim:
        print(os.environ[path])
        print(value)
        os.environ[path] = value + delim + os.environ[path]
        print(os.environ[path])
        print()
    else:
        os.environ[path] = value


def setenv(name, value):
    keys.add(name)
    os.environ[name] = value
    print(os.environ[name])
    print(value)
    print(os.environ[name])
    print()


def bruteforce():
    # gcc/8.4.0
    # cat /ssoft/spack/izar_stable/share/spack/lmod/izar/linux-rhel7-x86_64/Core/gcc/8.4.0.lua
    prepend_path("MODULEPATH", "/ssoft/spack/arvine/v1/share/spack/lmod/izar/linux-rhel7-x86_64/gcc/8.4.0")
    prepend_path("PATH", "/ssoft/spack/arvine/v1/opt/spack/linux-rhel7-haswell/gcc-4.8.5/gcc-8.4.0-fgpbrrd26pxv56imea5tnqj67vxh3a64/bin", ":")
    prepend_path("MANPATH", "/ssoft/spack/arvine/v1/opt/spack/linux-rhel7-haswell/gcc-4.8.5/gcc-8.4.0-fgpbrrd26pxv56imea5tnqj67vxh3a64/share/man", ":")
    prepend_path("LD_LIBRARY_PATH", "/ssoft/spack/arvine/v1/opt/spack/linux-rhel7-haswell/gcc-4.8.5/gcc-8.4.0-fgpbrrd26pxv56imea5tnqj67vxh3a64/lib", ":")
    prepend_path("LIBRARY_PATH", "/ssoft/spack/arvine/v1/opt/spack/linux-rhel7-haswell/gcc-4.8.5/gcc-8.4.0-fgpbrrd26pxv56imea5tnqj67vxh3a64/lib", ":")
    prepend_path("LD_LIBRARY_PATH", "/ssoft/spack/arvine/v1/opt/spack/linux-rhel7-haswell/gcc-4.8.5/gcc-8.4.0-fgpbrrd26pxv56imea5tnqj67vxh3a64/lib64", ":")
    prepend_path("LIBRARY_PATH", "/ssoft/spack/arvine/v1/opt/spack/linux-rhel7-haswell/gcc-4.8.5/gcc-8.4.0-fgpbrrd26pxv56imea5tnqj67vxh3a64/lib64", ":")
    prepend_path("C_INCLUDE_PATH", "/ssoft/spack/arvine/v1/opt/spack/linux-rhel7-haswell/gcc-4.8.5/gcc-8.4.0-fgpbrrd26pxv56imea5tnqj67vxh3a64/include", ":")
    prepend_path("CPLUS_INCLUDE_PATH", "/ssoft/spack/arvine/v1/opt/spack/linux-rhel7-haswell/gcc-4.8.5/gcc-8.4.0-fgpbrrd26pxv56imea5tnqj67vxh3a64/include", ":")
    prepend_path("INCLUDE", "/ssoft/spack/arvine/v1/opt/spack/linux-rhel7-haswell/gcc-4.8.5/gcc-8.4.0-fgpbrrd26pxv56imea5tnqj67vxh3a64/include", ":")
    prepend_path("CMAKE_PREFIX_PATH", "/ssoft/spack/arvine/v1/opt/spack/linux-rhel7-haswell/gcc-4.8.5/gcc-8.4.0-fgpbrrd26pxv56imea5tnqj67vxh3a64/", ":")
    setenv("CC", "/ssoft/spack/arvine/v1/opt/spack/linux-rhel7-haswell/gcc-4.8.5/gcc-8.4.0-fgpbrrd26pxv56imea5tnqj67vxh3a64/bin/gcc")
    setenv("CXX", "/ssoft/spack/arvine/v1/opt/spack/linux-rhel7-haswell/gcc-4.8.5/gcc-8.4.0-fgpbrrd26pxv56imea5tnqj67vxh3a64/bin/g++")
    setenv("FC", "/ssoft/spack/arvine/v1/opt/spack/linux-rhel7-haswell/gcc-4.8.5/gcc-8.4.0-fgpbrrd26pxv56imea5tnqj67vxh3a64/bin/gfortran")
    setenv("F77", "/ssoft/spack/arvine/v1/opt/spack/linux-rhel7-haswell/gcc-4.8.5/gcc-8.4.0-fgpbrrd26pxv56imea5tnqj67vxh3a64/bin/gfortran")
    setenv("GCC_ROOT", "/ssoft/spack/arvine/v1/opt/spack/linux-rhel7-haswell/gcc-4.8.5/gcc-8.4.0-fgpbrrd26pxv56imea5tnqj67vxh3a64")

    # cuda/10.1.243
    # cat /ssoft/spack/arvine/v1/share/spack/lmod/izar/linux-rhel7-x86_64/gcc/8.4.0/cuda/10.1.243.lua
    prepend_path("PATH", "/ssoft/spack/arvine/v1/opt/spack/linux-rhel7-skylake_avx512/gcc-8.4.0/cuda-10.1.243-ybyd6ny633ncbwclfv3qez63nyxzeby7/bin", ":")
    prepend_path("LD_LIBRARY_PATH", "/ssoft/spack/arvine/v1/opt/spack/linux-rhel7-skylake_avx512/gcc-8.4.0/cuda-10.1.243-ybyd6ny633ncbwclfv3qez63nyxzeby7/lib64", ":")
    prepend_path("LIBRARY_PATH", "/ssoft/spack/arvine/v1/opt/spack/linux-rhel7-skylake_avx512/gcc-8.4.0/cuda-10.1.243-ybyd6ny633ncbwclfv3qez63nyxzeby7/lib64", ":")
    prepend_path("C_INCLUDE_PATH", "/ssoft/spack/arvine/v1/opt/spack/linux-rhel7-skylake_avx512/gcc-8.4.0/cuda-10.1.243-ybyd6ny633ncbwclfv3qez63nyxzeby7/include", ":")
    prepend_path("CPLUS_INCLUDE_PATH", "/ssoft/spack/arvine/v1/opt/spack/linux-rhel7-skylake_avx512/gcc-8.4.0/cuda-10.1.243-ybyd6ny633ncbwclfv3qez63nyxzeby7/include", ":")
    prepend_path("INCLUDE", "/ssoft/spack/arvine/v1/opt/spack/linux-rhel7-skylake_avx512/gcc-8.4.0/cuda-10.1.243-ybyd6ny633ncbwclfv3qez63nyxzeby7/include", ":")
    prepend_path("CMAKE_PREFIX_PATH", "/ssoft/spack/arvine/v1/opt/spack/linux-rhel7-skylake_avx512/gcc-8.4.0/cuda-10.1.243-ybyd6ny633ncbwclfv3qez63nyxzeby7/", ":")
    setenv("CUDA_HOME", "/ssoft/spack/arvine/v1/opt/spack/linux-rhel7-skylake_avx512/gcc-8.4.0/cuda-10.1.243-ybyd6ny633ncbwclfv3qez63nyxzeby7")
    setenv("CUDA_ROOT", "/ssoft/spack/arvine/v1/opt/spack/linux-rhel7-skylake_avx512/gcc-8.4.0/cuda-10.1.243-ybyd6ny633ncbwclfv3qez63nyxzeby7")
    setenv("CUDA_LIBRARY", "/ssoft/spack/arvine/v1/opt/spack/linux-rhel7-skylake_avx512/gcc-8.4.0/cuda-10.1.243-ybyd6ny633ncbwclfv3qez63nyxzeby7/lib64")
    setenv("CUDA_INCLUDE", "/ssoft/spack/arvine/v1/opt/spack/linux-rhel7-skylake_avx512/gcc-8.4.0/cuda-10.1.243-ybyd6ny633ncbwclfv3qez63nyxzeby7/include")


if __name__ == '__main__':
    bruteforce()
    print(keys)
    for key in keys:
        print(f'{key}="{os.environ[key]}"')
    # init_lmod_modules(["gcc/8.4.0-cuda", "cuda/10.1"])
