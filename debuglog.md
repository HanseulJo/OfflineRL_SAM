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
