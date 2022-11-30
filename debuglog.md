# Debugging Log

## 1. Monitoring 오류

```bash
name 'glPushMatrix' is not defined
```

시도해본 것:

```python
from gym.wrappers import Monitor
from gym.wrappers.record_video import RecordVideo
```

예상되는 오류 지점: `pyglet.gl.gl.glPushMatrix` 가 어떤 이유에서인지 안 읽히고 있음.

***

## 2. Gym Pendulum 오류

```bash
DeprecatedEnv: Env Pendulum-v0 not found (valid versions include ['Pendulum-v1'])
```

`d3rlpy.datasets.py` 에서 `get_pendulum` 함수 수정하여 해결

```python
# environment
# env = gym.make("Pendulum-v0")
env = gym.make("Pendulum-v1")
```

## 3. Tensorboard directory 수정

중간에 `runs` directory가 생기는 현상 방지용으로 `logger.py`의 line 80즈음부터 수정

```python
if tensorboard_dir:
    tfboard_path = os.path.join(
        tensorboard_dir, self._experiment_name  # modified line
    )
    self._writer = SummaryWriter(logdir=tfboard_path)
```

## 4. OpenGL.framework 파일이 안 열리는 오류

Conda 환경에서 실행하면 glPushMatrix 오류가 계속되어서 global env로 변경했는데 아래와 같은 오류 발생.

```bash
Can't find framework /System/Library/frameworks/OpenGL.framework
```

`os.path.exists` method가 해당 폴더 내의 OpenGL을 찾지 못해 생긴 일.
`(python path)/pyglet/lib.py`의 `find_framework` method에 다음 코드라인을 넣어 해결. 그러나...

```python
### Added 221129 ###
if name in os.listdir(path):
    return realpath
####################
```

여전히 아래와 같은 문제 발생중.

```bash
File /opt/homebrew/lib/python3.10/site-packages/pyglet/libs/darwin/cocoapy/runtime.py:133
    130 objc.class_getMethodImplementation.argtypes = [c_void_p, c_void_p]
    132 # IMP class_getMethodImplementation_stret(Class cls, SEL name)
--> 133 objc.class_getMethodImplementation_stret.restype = c_void_p
    134 objc.class_getMethodImplementation_stret.argtypes = [c_void_p, c_void_p]
    136 # const char * class_getName(Class cls)

File /opt/homebrew/Cellar/python@3.10/3.10.7/Frameworks/Python.framework/Versions/3.10/lib/python3.10/ctypes/__init__.py:387, in CDLL.__getattr__(self, name)
    385 if name.startswith('__') and name.endswith('__'):
    386     raise AttributeError(name)
--> 387 func = self.__getitem__(name)
    388 setattr(self, name, func)
    389 return func

File /opt/homebrew/Cellar/python@3.10/3.10.7/Frameworks/Python.framework/Versions/3.10/lib/python3.10/ctypes/__init__.py:392, in CDLL.__getitem__(self, name_or_ordinal)
    391 def __getitem__(self, name_or_ordinal):
--> 392     func = self._FuncPtr((name_or_ordinal, self))
    393     if not isinstance(name_or_ordinal, int):
    394         func.__name__ = name_or_ordinal

AttributeError: dlsym(0x3a0a7cb68, class_getMethodImplementation_stret): symbol not found
```

## 5. offline RL 패키지 complie 중 gcc 관련 오류

각종 package 설치 후, `import d4rl` 혹은 `import mujoco_py` 실행하면 다음과 같은 오류 뜸:

```bash
RuntimeError: Could not find supported GCC executable.

HINT: On OS X, install GCC 9.x with `brew install gcc@9`. or `port install gcc9`.
```

M1 Mac에서 GCC-9를 지원하지 않는 것을 확인, GCC 버전 업그레이드 (`brew install gcc`) --> GCC 12 설치 완료. 아래처럼 환경변수 변경:

```python3
import os
os.environ['CC']="/opt/homebrew/bin/gcc-12"
```

혹은, bash에서

```bash
export CC=/opt/homebrew/bin/gcc-12
```

이제 위와 같은 오류 사라짐. 그러나 다시 import를 시행하면 아래와 같은 오류 발생.

```bash
$ python3 -c 'import d4rl'          
running build_ext
building 'mujoco_py.cymj' extension
/opt/homebrew/bin/gcc-12 -Wno-unused-result -Wsign-compare -Wunreachable-code -DNDEBUG -fwrapv -O2 -Wall -fPIC -O2 -isystem /opt/homebrew/Caskroom/miniforge/base/envs/RL/include -arch arm64 -fPIC -O2 -isystem /opt/homebrew/Caskroom/miniforge/base/envs/RL/include -arch arm64 -DONMAC -I/opt/homebrew/Caskroom/miniforge/base/envs/RL/lib/python3.10/site-packages/mujoco_py -I/Users/hanseul_jo/.mujoco/mujoco210/include -I/opt/homebrew/Caskroom/miniforge/base/envs/RL/lib/python3.10/site-packages/numpy/core/include -I/opt/homebrew/Caskroom/miniforge/base/envs/RL/include/python3.10 -c /opt/homebrew/Caskroom/miniforge/base/envs/RL/lib/python3.10/site-packages/mujoco_py/cymj.c -o /opt/homebrew/Caskroom/miniforge/base/envs/RL/lib/python3.10/site-packages/mujoco_py/generated/_pyxbld_2.1.2.14_310_macextensionbuilder/temp.macosx-11.1-arm64-cpython-310/opt/homebrew/Caskroom/miniforge/base/envs/RL/lib/python3.10/site-packages/mujoco_py/cymj.o -fopenmp -w
/opt/homebrew/Caskroom/miniforge/base/envs/RL/lib/python3.10/site-packages/mujoco_py/cymj.c: In function '__pyx_pf_9mujoco_py_4cymj_12PyMjrContext_15glewInitialized___get__':
/opt/homebrew/Caskroom/miniforge/base/envs/RL/lib/python3.10/site-packages/mujoco_py/cymj.c:84620:55: error: 'mjrContext' {aka 'struct mjrContext_'} has no member named 'glewInitialized'; did you mean 'glInitialized'?
84620 |   __pyx_t_1 = __Pyx_PyInt_From_int(__pyx_v_self->ptr->glewInitialized); if (unlikely(!__pyx_t_1)) __PYX_ERR(4, 4175, __pyx_L1_error)
      |                                                       ^~~~~~~~~~~~~~~
      |                                                       glInitialized
/opt/homebrew/Caskroom/miniforge/base/envs/RL/lib/python3.10/site-packages/mujoco_py/cymj.c: In function '__pyx_pf_9mujoco_py_4cymj_12PyMjrContext_15glewInitialized_2__set__':
/opt/homebrew/Caskroom/miniforge/base/envs/RL/lib/python3.10/site-packages/mujoco_py/cymj.c:84675:22: error: 'mjrContext' {aka 'struct mjrContext_'} has no member named 'glewInitialized'; did you mean 'glInitialized'?
84675 |   __pyx_v_self->ptr->glewInitialized = __pyx_v_x;
      |                      ^~~~~~~~~~~~~~~
      |                      glInitialized
Traceback (most recent call last):
  File "/opt/homebrew/Caskroom/miniforge/base/envs/RL/lib/python3.10/site-packages/setuptools/_distutils/unixccompiler.py", line 186, in _compile
    self.spawn(compiler_so + cc_args + [src, '-o', obj] + extra_postargs)
  File "/opt/homebrew/Caskroom/miniforge/base/envs/RL/lib/python3.10/site-packages/setuptools/_distutils/ccompiler.py", line 1007, in spawn
    spawn(cmd, dry_run=self.dry_run, **kwargs)
  File "/opt/homebrew/Caskroom/miniforge/base/envs/RL/lib/python3.10/site-packages/setuptools/_distutils/spawn.py", line 70, in spawn
    raise DistutilsExecError(
distutils.errors.DistutilsExecError: command '/opt/homebrew/bin/gcc-12' failed with exit code 1

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/opt/homebrew/Caskroom/miniforge/base/envs/RL/lib/python3.10/site-packages/d4rl/__init__.py", line 14, in <module>
    import d4rl.locomotion
  File "/opt/homebrew/Caskroom/miniforge/base/envs/RL/lib/python3.10/site-packages/d4rl/locomotion/__init__.py", line 2, in <module>
    from d4rl.locomotion import ant
  File "/opt/homebrew/Caskroom/miniforge/base/envs/RL/lib/python3.10/site-packages/d4rl/locomotion/ant.py", line 20, in <module>
    import mujoco_py
  File "/opt/homebrew/Caskroom/miniforge/base/envs/RL/lib/python3.10/site-packages/mujoco_py/__init__.py", line 2, in <module>
    from mujoco_py.builder import cymj, ignore_mujoco_warnings, functions, MujocoException
  File "/opt/homebrew/Caskroom/miniforge/base/envs/RL/lib/python3.10/site-packages/mujoco_py/builder.py", line 504, in <module>
    cymj = load_cython_ext(mujoco_path)
  File "/opt/homebrew/Caskroom/miniforge/base/envs/RL/lib/python3.10/site-packages/mujoco_py/builder.py", line 110, in load_cython_ext
    cext_so_path = builder.build()
  File "/opt/homebrew/Caskroom/miniforge/base/envs/RL/lib/python3.10/site-packages/mujoco_py/builder.py", line 226, in build
    built_so_file_path = self._build_impl()
  File "/opt/homebrew/Caskroom/miniforge/base/envs/RL/lib/python3.10/site-packages/mujoco_py/builder.py", line 343, in _build_impl
    so_file_path = super()._build_impl()
  File "/opt/homebrew/Caskroom/miniforge/base/envs/RL/lib/python3.10/site-packages/mujoco_py/builder.py", line 249, in _build_impl
    dist.run_commands()
  File "/opt/homebrew/Caskroom/miniforge/base/envs/RL/lib/python3.10/site-packages/setuptools/_distutils/dist.py", line 968, in run_commands
    self.run_command(cmd)
  File "/opt/homebrew/Caskroom/miniforge/base/envs/RL/lib/python3.10/site-packages/setuptools/dist.py", line 1217, in run_command
    super().run_command(command)
  File "/opt/homebrew/Caskroom/miniforge/base/envs/RL/lib/python3.10/site-packages/setuptools/_distutils/dist.py", line 987, in run_command
    cmd_obj.run()
  File "/opt/homebrew/Caskroom/miniforge/base/envs/RL/lib/python3.10/site-packages/Cython/Distutils/old_build_ext.py", line 186, in run
    _build_ext.build_ext.run(self)
  File "/opt/homebrew/Caskroom/miniforge/base/envs/RL/lib/python3.10/site-packages/setuptools/_distutils/command/build_ext.py", line 346, in run
    self.build_extensions()
  File "/opt/homebrew/Caskroom/miniforge/base/envs/RL/lib/python3.10/site-packages/mujoco_py/builder.py", line 149, in build_extensions
    build_ext.build_extensions(self)
  File "/opt/homebrew/Caskroom/miniforge/base/envs/RL/lib/python3.10/site-packages/Cython/Distutils/old_build_ext.py", line 195, in build_extensions
    _build_ext.build_ext.build_extensions(self)
  File "/opt/homebrew/Caskroom/miniforge/base/envs/RL/lib/python3.10/site-packages/setuptools/_distutils/command/build_ext.py", line 466, in build_extensions
    self._build_extensions_serial()
  File "/opt/homebrew/Caskroom/miniforge/base/envs/RL/lib/python3.10/site-packages/setuptools/_distutils/command/build_ext.py", line 492, in _build_extensions_serial
    self.build_extension(ext)
  File "/opt/homebrew/Caskroom/miniforge/base/envs/RL/lib/python3.10/site-packages/setuptools/_distutils/command/build_ext.py", line 547, in build_extension
    objects = self.compiler.compile(
  File "/opt/homebrew/Caskroom/miniforge/base/envs/RL/lib/python3.10/site-packages/setuptools/_distutils/ccompiler.py", line 599, in compile
    self._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)
  File "/opt/homebrew/Caskroom/miniforge/base/envs/RL/lib/python3.10/site-packages/setuptools/_distutils/unixccompiler.py", line 188, in _compile
    raise CompileError(msg)
distutils.errors.CompileError: command '/opt/homebrew/bin/gcc-12' failed with exit code 1
```
